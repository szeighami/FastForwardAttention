#include<limits>
#include<cmath>
#include<string>
#include<vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

float get_alph_c(int batch_size, int tpb, int tpv, int tpk, int prompt_len, int head_size)
{
	std::vector<float> features  = {(float)batch_size, (float)tpb, (float)tpv, (float)tpk, (float)prompt_len, (float)head_size, 1.0f/(float)batch_size, 1.0f/(float)tpb, 1.0f/(float)tpv, 1.0f/(float)tpk, 1.0f/(float)prompt_len, 1.0f/(float)head_size};

	//parameters equation found by fitting polynomial
	std::vector<float> coefs = {1.98477462e+00, 1.40485848e+00, 9.44788188e+01, 4.97399678e-01,
							  1.38083739e+01, 1.04151458e+00, 7.80703368e+01, 2.47025323e+03,
							  7.96088089e-01, 1.17568866e+02, 3.81078416e-02, 5.98652327e-02,
							  1.73769284e-04, 1.89844292e-02, 2.43233896e-01, 1.84552851e+01,
							  9.63335902e-03, 3.43894758e-01, 2.35635243e+02, 5.69168616e+02,
							  6.42865539e-01, 2.40778083e-02};

	std::vector<std::string> terms = {"6", "8", "11", "1 10", "3 7", "3 11", "6 11", "7 11", "8 9", "8 11", "0 1 10", "1 2 10", "1 3 6", "1 6 11", "1 8 11", "1 10 11", "2 3 6", "2 3 11", "2 10 11", "3 7 11", "5 7 8", "5 8 9"};

	float alpha_c = 0;
	for (uint32_t i = 0; i < coefs.size(); i++){
		size_t pos = 0;
		std::string token;
		std::string s = terms[i];
		float prod = 1;	
		while ((pos = s.find(" ")) != std::string::npos) {
			token = s.substr(0, pos);
			prod = prod*features[std::stoi(token)];
			s.erase(0, pos + 1);
		}
		alpha_c += prod*coefs[i];
	}
    
	return alpha_c;
}

int get_max_warps(int MyRegCount, int tpb, int head_size, int tpv, int precision, int prompt_len)
{
    int device = 0;
    if(getenv("CUDA_VISIBLE_DEVICES")) {
        std::stringstream ss(std::getenv("CUDA_VISIBLE_DEVICES")); ss >> device; 
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int limitRegsPerBlock = prop.regsPerBlock;
    int max_sharedmemory_per_block=prop.sharedMemPerBlock;
    int compute_capacity = prop.major;
    //Values obtained from nvidia occupancy calculator
    int registerAllocationUnitSize = 256 ;
    int myWarpAllocationGranularity = 4;
    int mySharedMemAllocationSize=compute_capacity >= 8 ? 128 : 256; 
    int run_time_smem=compute_capacity >= 8 ? 1024 : 0;

    int limitThreadsPerWarp = 32;
    int MyWarpsPerBlock = tpb/limitThreadsPerWarp;
    int smem_per_block = std::max((tpb/(head_size/tpv))*head_size*(precision/8)/2.0, std::ceil((prompt_len+1.0)/ 4.0) * 16+std::ceil((prompt_len+1.0)/ 4.0) * 4.0 * (precision/8))+run_time_smem;

    int reg_count_per_warp = std::ceil(MyRegCount*limitThreadsPerWarp/(float)registerAllocationUnitSize)*registerAllocationUnitSize;
    int reg_limit_sm = std::floor((limitRegsPerBlock/reg_count_per_warp)/(float)myWarpAllocationGranularity)*myWarpAllocationGranularity;
    int registers_max_warps = (reg_limit_sm/MyWarpsPerBlock)*MyWarpsPerBlock;

    int smem_used = std::ceil(smem_per_block/(float)mySharedMemAllocationSize)*mySharedMemAllocationSize;
    int smem_max_warps = (max_sharedmemory_per_block/smem_used)*MyWarpsPerBlock;

    int max_warps_per_sm = std::min(registers_max_warps, smem_max_warps);

    return max_warps_per_sm;
}

int get_k_size(int tpk)
{

    if (tpk == 1)
        return 8;
    if (tpk == 2)
        return 4;
    if (tpk == 4)
        return 2;
    printf("tpk not found");
    return -1;
}

float get_time_for_param(int precision, int batch_size, int head_num, int head_size, int prompt_len, int tpb, int tpv, int tpk)
{
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int no_sms = prop.multiProcessorCount;
    float gpu_clock_rate = prop.clockRate*1e3f;  
    float gpu_transfer_rate = 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 ;

	float mem_trasfer_bytes_calc = (precision/8) * batch_size * 4 * head_num*head_size+2*(precision/8)*batch_size*head_num*head_size*(prompt_len);

	int no_blocks = batch_size*head_num;
	int max_blocks_per_smp = std::ceil(no_blocks/(float)no_sms);
	int max_launched_warps = max_blocks_per_smp*(tpb/32);

	int MyRegCount_est;

	//parameters equation found by fitting polynomial
	if (precision == 16)
		MyRegCount_est = 93.61111111 + 0.44444444*head_size -18.61111111*tpk;
	else
		MyRegCount_est = 135.83333333 +0.96527778*head_size -38.54761905*tpk;

	int max_warps_per_sm_est = get_max_warps(MyRegCount_est, tpb,  head_size, tpv, precision, prompt_len);

	float overload_factor_est = 1;
	if (max_launched_warps>max_warps_per_sm_est)
		overload_factor_est = max_launched_warps/max_warps_per_sm_est;


	float alpha_m = 1;	
	float alpha_c = get_alph_c(batch_size, tpb, tpv, tpk, prompt_len, head_size);
   
	float time_mem_est = alpha_m*mem_trasfer_bytes_calc/(gpu_transfer_rate);

	int actual_tpv = head_size/tpv;
	//parameters equation found by fitting polynomial
    float no_inst_est;
	if (precision == 16)
		no_inst_est = 399.87009804 + 13.44339289*prompt_len*actual_tpv/tpb + -158.63911938*prompt_len*tpk/tpb +1.19653041*prompt_len*head_size/tpb+30.93940003*prompt_len*head_size/(tpb*get_k_size(tpk))+7.25426324*prompt_len/tpb;
	else
		no_inst_est = 450.37282135l + 1.30064347*prompt_len*actual_tpv/tpb + -64.04635652*prompt_len*tpk/tpb +2.34498193*prompt_len*head_size/tpb+49.51406228*prompt_len*head_size/(tpb*get_k_size(tpk))+60.20351018*prompt_len/tpb;

    float C = 3;
	float est_time = time_mem_est+(alpha_c*no_inst_est)/(gpu_clock_rate*std::pow(10,9))*overload_factor_est+C;

	return est_time;
}


void get_launch_params(int precision, int batch_size, int head_num, int head_size, int prompt_len, int& tpb, int& tpv, int& tpk)
{
	std::vector<int> tpks = {1, 2, 4};
	std::vector<int> tpbs = {64, 128, 256};
	std::vector<int> tpvs = {2, 4};
	if (precision == 16)
        tpvs.push_back(8);
    else
        tpvs.push_back(1);

    float min_time = std::numeric_limits<float>::max();
    for (auto curr_tpv: tpvs){
        for (auto curr_tpk: tpks){
            for (auto curr_tpb: tpbs){
                float time = get_time_for_param(precision, batch_size, head_num, head_size, prompt_len, curr_tpb, curr_tpv, curr_tpk);
                if (time < min_time){
                    time =  min_time;
                    tpb = curr_tpb;
                    tpk = curr_tpk;
                    tpv = curr_tpv;
                }
            }
        }
    }
}



/*
template<typename T>
void call_attention(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb);//{
    //printf("unsupported attention type");
//}

template<>
void call_attention<float>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, float* q, float* k, float* v, float* k_cache, float* v_cache, float* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb);

template<>
void call_attention<uint16_t>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, uint16_t* out, int* seq_lengths, float prob_thresh, int tpv, int tpk, int tpb);

template<typename T>
void do_attention(int batch_size,int head_num, int head_dim, int max_seq_length, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out, float softmax_scale, int curr_timestep, int* seq_lengths, float prob_thresh){

    int tpv; int tpk; int tpb;
    int precision = sizeof(T) == 4 ? 32 : 16;

    int no_sms = 82;  float C = 3;   float gpu_clock_rate = 1395e6; int mySharedMemAllocationSize=128; int max_sharedmemory_per_block=102400; int run_time_smem=1024;
    float gpu_transfer_rate =936*std::pow(2,30) ;

    get_launch_params(precision,batch_size, head_num, head_dim, max_seq_length, no_sms, C, gpu_transfer_rate, gpu_clock_rate, mySharedMemAllocationSize, max_sharedmemory_per_block, run_time_smem, tpb, tpv, tpk);

    call_attention<T>(batch_size, head_num, head_dim, max_seq_length, softmax_scale, curr_timestep, q, k, v, k_cache, v_cache, out, seq_lengths, prob_thresh, tpv, tpk, tpb);
}
*/
