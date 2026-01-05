#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_streamofblocks.h>
#include <math.h>
#include <stdint.h>
#include "hls_task.h"
using namespace std;

extern "C" {

void producer_0(
  hls::stream< float >& v0,
  hls::stream< float >& v1
) {	
  v1.write(v0.read());
}

void consumer_0(
  hls::stream< float >& v6,
  hls::stream< float >& v7
) {	
  v6.write(v7.read() + (float)1.000000);
}

void load_buf0(
  float v13[16], hls::stream<float> &s1
) {	//
  l_S_load_buf0_load_buf0_l_0: for (int load_buf0_l_0 = 0; load_buf0_l_0 < 16; load_buf0_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v16 = v13[load_buf0_l_0];	//
    s1.write(v16);
  }
}

void store_res1(
  hls::stream<float> &s2, float v18[16]
) {	//
  l_S_store_res1_store_res1_l_0: for (int store_res1_l_0 = 0; store_res1_l_0 < 16; store_res1_l_0++) {	//
  #pragma HLS pipeline II=1 rewind
    float v20 = s2.read();	//
    v18[store_res1_l_0] = v20;	//
  }
}

void top(
  float *v21,
  float *v22
) {	// L24
  #pragma HLS interface m_axi port=v21 offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=v22 offset=slave bundle=gmem1
  #pragma HLS dataflow
  hls_thread_local hls::stream<float> sk1;
  hls_thread_local hls::stream<float> sk2;
  load_buf0(v21, sk1);	//
  hls::stream< float > v25;
  #pragma HLS stream variable=v25 depth=4	// L25
  hls_thread_local hls::task t1(producer_0, sk1, v25);	// L26
  hls_thread_local hls::task t2(consumer_0, sk2, v25);	// L27
  store_res1(sk2, v22);	//
}

} // extern "C"
