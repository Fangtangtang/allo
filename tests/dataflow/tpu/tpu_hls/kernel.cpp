
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_streamofblocks.h>
#include <math.h>
#include <stdint.h>
using namespace std;

extern "C" {

void DECODER_0(
  uint32_t v0[5],
  hls::stream< uint8_t >& v1,
  hls::stream< uint8_t >& v2,
  hls::stream< uint8_t >& v3,
  hls::stream< uint8_t >& v4,
  hls::stream< uint8_t >& v5,
  hls::stream< uint8_t >& v6,
  hls::stream< uint8_t >& v7,
  hls::stream< uint8_t >& v8,
  hls::stream< uint8_t >& v9,
  hls::stream< uint8_t >& v10,
  hls::stream< uint8_t >& v11,
  hls::stream< uint8_t >& v12,
  hls::stream< uint8_t >& v13,
  hls::stream< uint8_t >& v14,
  hls::stream< uint8_t >& v15
) {	// L2
  uint8_t nop;	// L19
  nop = 0;	// L20
  l_S_i_0_i: for (int i = 0; i < 5; i++) {	// L21
    uint32_t v18 = v0[(i)];	// L22
    uint32_t inst;	// L23
    inst = v18;	// L24
    int32_t v20 = inst;	// L25
    int8_t v21;
    ap_int<32> v21_tmp = v20;
    v21 = v21_tmp(27, 20);	// L26
    uint8_t rs1_addr;	// L27
    rs1_addr = v21;	// L28
    int32_t v23 = inst;	// L29
    int8_t v24;
    ap_int<32> v24_tmp = v23;
    v24 = v24_tmp(19, 12);	// L30
    uint8_t rs2_addr;	// L31
    rs2_addr = v24;	// L32
    int32_t v26 = inst;	// L33
    int8_t v27;
    ap_int<32> v27_tmp = v26;
    v27 = v27_tmp(11, 4);	// L34
    uint8_t rd_addr;	// L35
    rd_addr = v27;	// L36
    int32_t v29 = inst;	// L37
    ap_int<4> v30;
    ap_int<32> v30_tmp = v29;
    v30 = v30_tmp(3, 0);	// L38
    ap_uint<4> opcode;	// L39
    opcode = v30;	// L40
    ap_int<4> v32 = opcode;	// L41
    int32_t v33 = v32;	// L42
    bool v34 = v33 == 1;	// L43
    if (v34) {	// L44
      ap_int<4> v35 = opcode;	// L45
      uint8_t v36 = v35;	// L46
      uint8_t mem_inst_;	// L47
      mem_inst_ = v36;	// L48
      int8_t v38 = mem_inst_;	// L49
      v1.write(v38);	// L50
      int8_t v39 = rs1_addr;	// L51
      v2.write(v39);	// L52
      int8_t v40 = rs2_addr;	// L53
      v3.write(v40);	// L54
      int8_t v41 = nop;	// L55
      v4.write(v41);	// L56
    } else {
      ap_int<4> v42 = opcode;	// L58
      int32_t v43 = v42;	// L59
      bool v44 = v43 == 3;	// L60
      bool v45 = v43 == 4;	// L63
      bool v46 = v44 | v45;	// L64
      if (v46) {	// L65
        ap_int<4> v47 = opcode;	// L66
        uint8_t v48 = v47;	// L67
        uint8_t mem_inst_1;	// L68
        mem_inst_1 = v48;	// L69
        int8_t v50 = mem_inst_1;	// L70
        v5.write(v50);	// L71
        int8_t v51 = rs1_addr;	// L72
        v6.write(v51);	// L73
        int8_t v52 = rd_addr;	// L74
        v7.write(v52);	// L75
        int8_t v53 = nop;	// L76
        v8.write(v53);	// L77
      } else {
        ap_int<4> v54 = opcode;	// L79
        int32_t v55 = v54;	// L80
        bool v56 = v55 == 2;	// L81
        if (v56) {	// L82
          ap_int<4> v57 = opcode;	// L83
          uint8_t v58 = v57;	// L84
          uint8_t mem_inst_2;	// L85
          mem_inst_2 = v58;	// L86
          int8_t v60 = mem_inst_2;	// L87
          v9.write(v60);	// L88
          int8_t v61 = nop;	// L89
          v10.write(v61);	// L90
        } else {
          ap_int<4> v62 = opcode;	// L92
          int32_t v63 = v62;	// L93
          bool v64 = v63 == 8;	// L94
          if (v64) {	// L95
            uint8_t mem_inst_3;	// L96
            mem_inst_3 = 5;	// L97
            int8_t v66 = mem_inst_3;	// L98
            v11.write(v66);	// L99
            int8_t v67 = rs1_addr;	// L100
            v12.write(v67);	// L101
            int8_t v68 = rs2_addr;	// L102
            v13.write(v68);	// L103
            uint8_t vec_inst_;	// L104
            vec_inst_ = 2;	// L105
            int8_t v70 = vec_inst_;	// L106
            v14.write(v70);	// L107
            int8_t v71 = rd_addr;	// L108
            v15.write(v71);	// L109
          }
        }
      }
    }
  }
}

void MMU_0(
  ap_uint<256> v72[32],
  ap_uint<256> v73[32],
  hls::stream< uint8_t >& v74,
  hls::stream< uint8_t >& v75,
  hls::stream< uint8_t >& v76,
  hls::stream< uint8_t >& v77,
  hls::stream< uint8_t >& v78,
  hls::stream< uint8_t >& v79,
  hls::stream< uint8_t >& v80,
  hls::stream< uint8_t >& v81,
  hls::stream< uint8_t >& v82,
  hls::stream< ap_uint<256> >& v83,
  hls::stream< ap_uint<256> >& v84,
  hls::stream< uint8_t >& v85,
  hls::stream< ap_uint<256> >& v86
) {	// L117
  ap_uint<256> memory[32];	// L127
  ap_uint<4> bit_map[32];	// L128
  for (int v89 = 0; v89 < 32; v89++) {	// L129
    bit_map[v89] = 0;	// L129
  }
  ap_uint<2> vec_queue_bit_map[4];	// L130
  for (int v91 = 0; v91 < 4; v91++) {	// L131
    vec_queue_bit_map[v91] = 0;	// L131
  }
  l_S___0__: for (int _ = 0; _ < 5; _++) {	// L132
    uint8_t v93 = v74.read();	// L133
    uint8_t mem_inst_4;	// L134
    mem_inst_4 = v93;	// L135
    int8_t v95 = mem_inst_4;	// L136
    int32_t v96 = v95;	// L137
    bool v97 = v96 == 1;	// L138
    if (v97) {	// L139
      uint8_t v98 = v75.read();	// L140
      uint8_t starting_addr;	// L141
      starting_addr = v98;	// L142
      uint8_t v100 = v76.read();	// L143
      uint8_t size;	// L144
      size = v100;	// L145
      int8_t v102 = size;	// L146
      int v103 = v102;	// L147
      for (int v104 = 0; v104 < v103; v104 += 1) {	// L148
        int8_t v105 = starting_addr;	// L149
        ap_uint<33> v106 = v105;	// L150
        ap_uint<33> v107 = v104;	// L151
        ap_uint<33> v108 = v106 + v107;	// L152
        int v109 = v108;	// L153
        bit_map[v109] = 3;	// L154
      }
    } else {
      int8_t v110 = mem_inst_4;	// L157
      int32_t v111 = v110;	// L158
      bool v112 = v111 == 3;	// L159
      if (v112) {	// L160
        uint8_t v113 = v77.read();	// L161
        uint8_t src_addr;	// L162
        src_addr = v113;	// L163
        uint8_t v115 = v78.read();	// L164
        uint8_t dst_addr;	// L165
        dst_addr = v115;	// L166
        int8_t v117 = src_addr;	// L167
        int v118 = v117;	// L168
        ap_uint<256> v119 = v72[v118];	// L169
        int8_t v120 = dst_addr;	// L170
        int v121 = v120;	// L171
        memory[v121] = v119;	// L172
      } else {
        int8_t v122 = mem_inst_4;	// L174
        int32_t v123 = v122;	// L175
        bool v124 = v123 == 4;	// L176
        if (v124) {	// L177
          uint8_t v125 = v79.read();	// L178
          uint8_t src_addr1;	// L179
          src_addr1 = v125;	// L180
          uint8_t v127 = v80.read();	// L181
          uint8_t dst_addr1;	// L182
          dst_addr1 = v127;	// L183
          int8_t v129 = src_addr1;	// L184
          int v130 = v129;	// L185
          ap_uint<256> v131 = memory[v130];	// L186
          int8_t v132 = dst_addr1;	// L187
          int v133 = v132;	// L188
          v73[v133] = v131;	// L189
        } else {
          int8_t v134 = mem_inst_4;	// L191
          int32_t v135 = v134;	// L192
          bool v136 = v135 == 5;	// L193
          if (v136) {	// L194
            uint8_t v137 = v81.read();	// L195
            uint8_t vs1_addr;	// L196
            vs1_addr = v137;	// L197
            uint8_t v139 = v82.read();	// L198
            uint8_t vs2_addr;	// L199
            vs2_addr = v139;	// L200
            int8_t v141 = vs1_addr;	// L201
            int v142 = v141;	// L202
            ap_uint<4> v143 = bit_map[v142];	// L203
            int32_t v144 = v143;	// L204
            bool v145 = v144 == 3;	// L205
            if (v145) {	// L206
              int8_t v146 = vs1_addr;	// L207
              int v147 = v146;	// L208
              ap_uint<256> v148 = memory[v147];	// L209
              v83.write(v148);	// L210
            }
            int8_t v149 = vs2_addr;	// L212
            int v150 = v149;	// L213
            ap_uint<4> v151 = bit_map[v150];	// L214
            int32_t v152 = v151;	// L215
            bool v153 = v152 == 3;	// L216
            if (v153) {	// L217
              int8_t v154 = vs2_addr;	// L218
              int v155 = v154;	// L219
              ap_uint<256> v156 = memory[v155];	// L220
              v84.write(v156);	// L221
            }
            uint8_t v157 = v85.read();	// L223
            uint8_t vd_addr;	// L224
            vd_addr = v157;	// L225
            ap_uint<256> v159 = v86.read();	// L226
            ap_uint<256> vec_result;	// L227
            vec_result = v159;	// L228
            int8_t v161 = vd_addr;	// L229
            int v162 = v161;	// L230
            bit_map[v162] = 3;	// L231
            ap_int<256> v163 = vec_result;	// L232
            int8_t v164 = vd_addr;	// L233
            int v165 = v164;	// L234
            memory[v165] = v163;	// L235
          }
        }
      }
    }
  }
}

void VEC_0(
  hls::stream< uint8_t >& v166,
  hls::stream< ap_uint<256> >& v167,
  hls::stream< ap_uint<256> >& v168,
  hls::stream< ap_uint<256> >& v169
) {	// L243
  l_S___0__1: for (int _1 = 0; _1 < 5; _1++) {	// L249
    uint8_t v171 = v166.read();	// L250
    uint8_t vec_inst_1;	// L251
    vec_inst_1 = v171;	// L252
    int8_t v173 = vec_inst_1;	// L253
    int32_t v174 = v173;	// L254
    bool v175 = v174 == 2;	// L255
    if (v175) {	// L256
      ap_uint<256> v176 = v167.read();	// L257
      ap_uint<256> operand1;	// L258
      operand1 = v176;	// L259
      ap_uint<256> v178 = v168.read();	// L260
      ap_uint<256> operand2;	// L261
      operand2 = v178;	// L262
      ap_uint<256> add_8b;	// L263
      l_arith_8_i1: for (int i1 = 0; i1 < 32; i1++) {	// L264
        ap_int<256> v182 = operand1;	// L265
        int64_t v183 = i1;	// L266
        int64_t v184 = v183 * 8;	// L267
        ap_int<34> v185 = i1;	// L268
        ap_int<34> v186 = v185 + 1;	// L269
        ap_int<66> v187 = v186;	// L270
        ap_int<66> v188 = v187 * 8;	// L271
        ap_int<66> v189 = v188 - 1;	// L272
        int v190 = v184;	// L273
        int v191 = v189;	// L274
        int8_t v192;
        ap_int<256> v192_tmp = v182;
        v192 = v192_tmp(v191, v190);	// L275
        int8_t scalar1;	// L276
        scalar1 = v192;	// L277
        ap_int<256> v194 = operand2;	// L278
        int8_t v195;
        ap_int<256> v195_tmp = v194;
        v195 = v195_tmp(v191, v190);	// L288
        int8_t scalar2;	// L289
        scalar2 = v195;	// L290
        int8_t v197 = scalar2;	// L291
        int8_t v198 = scalar1;	// L292
        ap_int<9> v199 = v197;	// L293
        ap_int<9> v200 = v198;	// L294
        ap_int<9> v201 = v199 + v200;	// L295
        uint8_t v202 = v201;	// L296
        ap_int<256> v203 = add_8b;	// L297
        ap_int<256> v204;
        ap_int<256> v204_tmp = v203;
        v204_tmp(v191, v190) = v202;
        v204 = v204_tmp;	// L307
        add_8b = v204;	// L308
      }
      ap_int<256> v205 = add_8b;	// L310
      v169.write(v205);	// L311
    }
  }
}

void top(
  uint32_t *v206,
  ap_uint<256> *v207,
  ap_uint<256> *v208
) {	// L316
  #pragma HLS interface m_axi port=v206 offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=v207 offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=v208 offset=slave bundle=gmem2
  #pragma HLS dataflow
  hls::stream< uint8_t > v209;
  #pragma HLS stream variable=v209 depth=4	// L317
  hls::stream< uint8_t > v210;
  #pragma HLS stream variable=v210 depth=4	// L318
  hls::stream< uint8_t > v211;
  #pragma HLS stream variable=v211 depth=4	// L319
  hls::stream< uint8_t > v212;
  #pragma HLS stream variable=v212 depth=4	// L320
  hls::stream< ap_uint<256> > v213;
  #pragma HLS stream variable=v213 depth=4	// L321
  hls::stream< ap_uint<256> > v214;
  #pragma HLS stream variable=v214 depth=4	// L322
  hls::stream< ap_uint<256> > v215;
  #pragma HLS stream variable=v215 depth=4	// L323
  DECODER_0(v206, v209, v210, v211, v212, v209, v210, v211, v212, v209, v212, v209, v210, v211, v212, v211);	// L324
  MMU_0(v207, v208, v209, v210, v211, v210, v211, v210, v211, v210, v211, v213, v214, v211, v215);	// L325
  VEC_0(v212, v213, v214, v215);	// L326
}


} // extern "C"
