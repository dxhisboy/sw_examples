#include <slave.h>
#include <simd.h>
#include <dma.h>
#include "param.h"
#define CPE_CNT 64
#define CPE_ROW 512
/*0x44=01 00 01 00, in1[1], in1[0], in0[1], in0[0]*/
/*0xEE=11 10 11 10, in1[3], in1[2], in0[3], in0[2]*/
/*0x88=10 00 10 00, in1[2], in1[0], in0[2], in0[0]*/
/*0xDD=11 01 11 01, in1[3], in1[1], in0[3], in0[1]*/

/* a  3  2  1  0 */
/* 0 03 02 01 00 */
/* 1 13 12 11 10 */
/* 2 23 22 21 20 */
/* 3 33 32 31 30 */

/* shff([13 12 11 10], [03 02 01 00], 0x44) -> 11 10 01 00 */
/* shff([13 12 11 10], [03 02 01 00], 0xEE) -> 03 02 13 12 */
/* shff([33 32 31 30], [23 22 21 20], 0x44) -> 31 30 21 20 */
/* shff([33 32 31 30], [23 22 21 20], 0xEE) -> 33 32 23 22 */

/* shff([31 30 21 20], [11 10 01 00], 0x88) -> 30 20 10 00 */
/* shff([31 30 21 20], [11 10 01 00], 0xDD) -> 31 21 11 01 */
/* shff([33 32 23 22], [13 12 03 02], 0x88) -> 32 22 12 02 */
/* shff([33 32 23 22], [13 12 03 02], 0xDD) -> 33 23 13 03 */

#define transpose_4x4(in0, in1, in2, in3, ot0, ot1, ot2, ot3) {	\
    doublev4 o0 = simd_vshff(in1,in0,0x44);			\
    doublev4 o1 = simd_vshff(in1,in0,0xEE);			\
    doublev4 o2 = simd_vshff(in3,in2,0x44);			\
    doublev4 o3 = simd_vshff(in3,in2,0xEE);			\
    ot0 = simd_vshff(o2,o0,0x88);				\
    ot1 = simd_vshff(o2,o0,0xDD);				\
    ot2 = simd_vshff(o3,o1,0x88);				\
    ot3 = simd_vshff(o3,o1,0xDD);				\
  }			       

#define rtc(_x) asm volatile("rcsr %0, 4" : "=r"(_x))
void do_mm_slave(mm_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);
  doublev4 a_v4[CPE_ROW], b_v4[4], c_v4[CPE_ROW];
  double (*a)[4] = a_v4, (*b)[4] = b_v4, (*c)[4] = c_v4;
  my_id = athread_get_id(-1);

  mm_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(mm_param_t), &get_reply, 0, 0, 0);
  while (get_reply != 1);

  get_reply = 0;
  athread_get(PE_MODE, pm.b, b, sizeof(double) * 4 * 4, &get_reply, 0, 0, 0);
  while (get_reply != 1);

  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);
  long t_sum = 0;
  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    long t_start;
    rtc(t_start);
    int i, j, k;
    for (i = 0; i < r_cnt; i ++)
      for (j = 0; j < 4; j ++)
	c[i][j] = 0;
    for (i = 0; i < r_cnt; i ++)
      for (j = 0; j < 4; j ++)
	for (k = 0; k < 4; k ++)
	  c[i][j] += a[i][k] * b[k][j];
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;
    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * 4, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);
  if (my_id == 0){
    printf("Cycles for computation of do_mm_slave: %lld\n", t_sum);
    printf("Cycles for all parts of do_mm_slave: %lld\n", f_end - f_start);
  }
}

void do_mm_slave_loopo(mm_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);

  doublev4 a_v4[CPE_ROW], b_v4[4], c_v4[CPE_ROW];
  double (*a)[4] = a_v4, (*b)[4] = b_v4, (*c)[4] = c_v4;
  my_id = athread_get_id(-1);

  mm_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(mm_param_t), &get_reply, 0, 0, 0);
  while (get_reply != 1);

  get_reply = 0;
  athread_get(PE_MODE, pm.b, b, sizeof(double) * 4 * 4, &get_reply, 0, 0, 0);
  while (get_reply != 1);

  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);
  long t_sum = 0;
  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    long t_start;
    rtc(t_start);
    int i, j, k;
    for (i = 0; i < r_cnt; i ++)
      for (j = 0; j < 4; j ++)
	c[i][j] = 0;
    for (i = 0; i < r_cnt; i ++)
      for (k = 0; k < 4; k ++)
	for (j = 0; j < 4; j ++)
	  c[i][j] += a[i][k] * b[k][j];
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;
    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * 4, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);

  if (my_id == 0){
    printf("Cycles for computation of do_mm_slave_loopo: %lld\n", t_sum);
    printf("Cycles for all parts of do_mm_slave_loopo: %lld\n", f_end - f_start);
  }
}

void do_mm_slave_vec(mm_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);

  doublev4 a_v4[CPE_ROW], b_v4[4], c_v4[CPE_ROW];
  double (*a)[4] = a_v4, (*b)[4] = b_v4, (*c)[4] = c_v4;
  my_id = athread_get_id(-1);

  mm_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(mm_param_t), &get_reply, 0, 0, 0);
  while (get_reply != 1);

  get_reply = 0;
  athread_get(PE_MODE, pm.b, b, sizeof(double) * 4 * 4, &get_reply, 0, 0, 0);
  while (get_reply != 1);

  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);
  long t_sum = 0;
  doublev4 v4_0 = 0;
  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    long t_start;
    rtc(t_start);
    int i, j, k;
    for (i = 0; i < r_cnt; i ++){
      c_v4[i] = v4_0;
      for (k = 0; k < 4; k ++){
	doublev4 aik_v4 = a[i][k];
	c_v4[i] = c_v4[i] + aik_v4 * b_v4[k];
      }
    }
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;
    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * 4, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);

  if (my_id == 0) {
    printf("Cycles for computation of do_mm_slave_vec: %lld\n", t_sum);
    printf("Cycles for all parts of do_mm_slave_vec: %lld\n", f_end - f_start);
  }
}

void do_mm_slave_vec_loope(mm_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);

  doublev4 a_v4[CPE_ROW], b_v4[4], c_v4[CPE_ROW];
  double (*a)[4] = a_v4, (*b)[4] = b_v4, (*c)[4] = c_v4;
  my_id = athread_get_id(-1);

  mm_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(mm_param_t), &get_reply, 0, 0, 0);
  while (get_reply != 1);

  get_reply = 0;
  athread_get(PE_MODE, pm.b, b, sizeof(double) * 4 * 4, &get_reply, 0, 0, 0);
  while (get_reply != 1);

  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);
  long t_sum = 0;
  doublev4 v4_0 = 0;
  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    long t_start;
    rtc(t_start);
    int i, j, k;
    for (i = 0; i < r_cnt; i ++){
      c_v4[i] = v4_0;
      doublev4 ai0_v4 = simd_vshff(a_v4[i], a_v4[i], 0x00);
      doublev4 ai1_v4 = simd_vshff(a_v4[i], a_v4[i], 0x55);
      doublev4 ai2_v4 = simd_vshff(a_v4[i], a_v4[i], 0xaa);
      doublev4 ai3_v4 = simd_vshff(a_v4[i], a_v4[i], 0xff);

      c_v4[i] = c_v4[i] + ai0_v4 * b_v4[0];
      c_v4[i] = c_v4[i] + ai1_v4 * b_v4[1];
      c_v4[i] = c_v4[i] + ai2_v4 * b_v4[2];
      c_v4[i] = c_v4[i] + ai3_v4 * b_v4[3];
    }
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;
    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * 4, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);

  if (my_id == 0){
    printf("Cycles for computation of do_mm_slave_vec_loope: %lld\n", t_sum);
    printf("Cycles for all parts of do_mm_slave_vec_loope: %lld\n", f_end - f_start);
  }
}

void do_hop_slave(hop_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);

  doublev4 a_v4[CPE_ROW], c_v4[(CPE_ROW + 3) / 4];
  double (*a)[4] = a_v4, *c = c_v4;
  my_id = athread_get_id(-1);
  hop_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(hop_param_t), &get_reply, 0, 0, 0);

  while (get_reply != 1);
  asm volatile("memb");
  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);

  long t_sum = 0;
  doublev4 v4_0 = 0;
  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    long t_start;
    rtc(t_start);
    int i, j, k;
    for (i = 0; i < r_cnt; i ++){
      c[i] = a[i][0] / a[i][1] + a[i][2] / a[i][3];
    }
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;
    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);

  if (my_id == 0){
    printf("Cycles for computation of do_hop_slave: %lld\n", t_sum);
    printf("Cycles for all parts of do_hop_slave_vec: %lld\n", f_end - f_start);
  }
}

void do_hop_slave_vec_T(hop_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);

  doublev4 a_v4[CPE_ROW], c_v4[(CPE_ROW + 3) / 4];
  double (*a)[4] = a_v4, *c = c_v4;
  my_id = athread_get_id(-1);
  hop_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(hop_param_t), &get_reply, 0, 0, 0);

  while (get_reply != 1);
  asm volatile("memb");

  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);
  long t_sum = 0;
  doublev4 v4_0 = 0;
  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    long t_start;
    rtc(t_start);
    int i, j, k;
    doublev4 t0, t1, t2, t3;
    for (i = 0; i < r_cnt; i += 4){
      transpose_4x4(a_v4[i + 0],
		    a_v4[i + 1],
		    a_v4[i + 2],
		    a_v4[i + 3],
		    t0, t1, t2, t3);
      c_v4[i >> 2] = t0 / t1 + t2 / t3;
    }
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;
    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);

  if (my_id == 0){
    printf("Cycles for computation of do_hop_slave_vec_T: %lld\n", t_sum);
    printf("Cycles for all parts of do_hop_slave_vec_T: %lld\n", f_end - f_start);
  }
}


void do_accum_slave_vec(accum_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);

  doublev4 a_v4[CPE_ROW], c_v4[CPE_ROW];
  double (*a)[4] = a_v4, *c = c_v4;
  my_id = athread_get_id(-1);
  if (my_id > 0)
    return;

  accum_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(accum_param_t), &get_reply, 0, 0, 0);

  while (get_reply != 1);
  asm volatile("memb");

  int a_start = 0;
  int a_end = pm.size;
  long t_sum = 0;
  doublev4 v4_0 = 0;
  doublev4 last_c = 0;
  while (a_start < a_end){
    long t_start;
    rtc(t_start);

    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    asm volatile("memb");
    int i, j, k;
    doublev4 t0, t1, t2, t3;
    c_v4[0] = last_c + a_v4[0];
    for (i = 1; i < r_cnt; i ++){
      c_v4[i] = c_v4[i - 1] + a_v4[i];
    }
    last_c = c_v4[r_cnt - 1];
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;

    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * 4, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);

  if (my_id == 0){
    printf("Cycles for computation of do_accum_slave_vec: %lld\n", t_sum);
    printf("Cycles for all parts of do_accum_slave_vec: %lld\n", f_end - f_start);
  }
}

void do_accum_slave_vec_rc(accum_param_t *gl_pm){
  volatile int get_reply, put_reply;

  long f_start;
  rtc(f_start);

  doublev4 a_v4[CPE_ROW], c_v4[CPE_ROW];
  double (*a)[4] = a_v4, *c = c_v4;
  int my_id = athread_get_id(-1);
  int row = my_id / 8;
  int col = my_id % 8;

  int direction, firstcol, lastcol;
  if (row & 1)
    direction = -1;
  else
    direction = 1;
  if (row & 1)
    firstcol = 7;
  else
    firstcol = 0;

  lastcol = 7 - firstcol;

  int remap_id = col * direction + 8 * row + 7 * (row & 1);
  accum_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(accum_param_t), &get_reply, 0, 0, 0);

  while (get_reply != 1);
  asm volatile("memb");
  int a_start = pm.size / CPE_CNT * remap_id;
  int a_end = pm.size / CPE_CNT * (remap_id + 1);

  long t_sum = 0;
  doublev4 v4_0 = 0;
  doublev4 last_c = 0;
  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;

    get_reply = 0;
    athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    asm volatile("memb");
    long t_start;
    rtc(t_start);

    int i, j, k;
    doublev4 t0, t1, t2, t3;
    c_v4[0] = last_c + a_v4[0];
    for (i = 1; i < r_cnt; i ++){
      c_v4[i] = c_v4[i - 1] + a_v4[i];
    }
    last_c = c_v4[r_cnt - 1];

    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;

    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * 4, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }

  doublev4 prev_c = 0;
  long t_start;
  rtc(t_start);

  if (remap_id > 0){
    if (col == firstcol)
      asm volatile("getc %0" : "=r"(prev_c));
    else
      asm volatile("getr %0" : "=r"(prev_c));
  }
  /* printf("%d get:", my_id); */
  /* simd_print_doublev4(prev_c); */
  last_c = last_c + prev_c;

  /* printf("%d put:", my_id); */
  /* simd_print_doublev4(last_c); */
    
  if (remap_id != CPE_CNT){
    if (col == lastcol)
      simd_putc(last_c, row + 1);
    else
      simd_putr(last_c, col + direction);
  }
  long t_end;
  rtc(t_end);
  t_sum += t_end - t_start;

  a_start = pm.size / CPE_CNT * my_id;

  while (a_start < a_end){
    int r_cnt;
    if (a_end - a_start < CPE_ROW)
      r_cnt = a_end - a_start;
    else
      r_cnt = CPE_ROW;
    get_reply = 0;
    athread_get(PE_MODE, pm.c + a_start, c, sizeof(double) * r_cnt * 4, &get_reply, 0, 0, 0);
    while (get_reply != 1);
    asm volatile("memb");
    long t_start;
    rtc(t_start);

    int i, j, k;
    doublev4 t0, t1, t2, t3;
    for (i = 0; i < r_cnt; i ++){
      c_v4[i] = c_v4[i] + prev_c;
    }
    long t_end;
    rtc(t_end);
    t_sum += t_end - t_start;

    put_reply = 0;
    athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * 4, &put_reply, 0, 0);
    while (put_reply != 1);
    a_start += r_cnt;
  }
  long f_end;
  rtc(f_end);
  
  if (my_id == 0){
    printf("Cycles for computation of do_accum_slave_vec_rc: %lld\n", t_sum);
    printf("Cycles for all parts of do_accum_slave_vec_rc: %lld\n", f_end - f_start);
  }
}

#define FLIP_CPE_ROW 1
void do_flip_slave(flip_param_t *gl_pm){
  int my_id;
  volatile int get_reply, put_reply;
  long t_start;
  rtc(t_start);

  doublev4 a_v4[FLIP_CPE_ROW][FLIP_ROW_SIZE / 4], c_v4[FLIP_CPE_ROW][FLIP_ROW_SIZE / 4];
  double (*a)[FLIP_ROW_SIZE] = a_v4, (*c)[FLIP_ROW_SIZE] = c_v4;
  my_id = athread_get_id(-1);
  flip_param_t pm;
  get_reply = 0;
  athread_get(PE_MODE, gl_pm, &pm, sizeof(flip_param_t), &get_reply, 0, 0, 0);

  while (get_reply != 1);
  asm volatile("memb");
  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);
  int row = my_id / 8;
  int col = my_id % 8;
  long t_sum = 0;
  doublev4 v4_0 = 0;

  int r_cnt = 1;

  get_reply = 0;
  athread_get(PE_MODE, pm.a + a_start, a, sizeof(double) * r_cnt * FLIP_ROW_SIZE, &get_reply, 0, 0, 0);
  while (get_reply != 1);
  asm volatile("memb");
  int i, j, k;

  for (i = 0; i < r_cnt; i ++){
    for (j = 0; j < FLIP_ROW_SIZE; j ++)
      c[i][j] = a[i][FLIP_ROW_SIZE - 1 - j];
  }
  put_reply = 0;
  athread_put(PE_MODE, c, pm.c + a_start, sizeof(double) * r_cnt * FLIP_ROW_SIZE, &put_reply, 0, 0);
  while (put_reply != 1);
  a_start += r_cnt;

  athread_syn(ARRAY_SCOPE, 0xffff);
  long t_end;
  rtc(t_end);
  t_sum += t_end - t_start;

  if (my_id == 0)
    printf("Cycles for all parts of do_flip_slave: %lld\n", t_sum);
}

void do_flip_slave_dma(flip_param_t *gl_pm){
  int my_id;
  dma_desc get_desc, put_desc;
  volatile int get_reply, put_reply;
  long t_start;
  rtc(t_start);

  dma_set_op(&get_desc, DMA_GET);
  dma_set_reply(&get_desc, &get_reply);
  dma_set_stepsize(&get_desc, 0);

  dma_set_mask(&get_desc, 0);
  dma_set_mode(&get_desc, PE_MODE);
  dma_set_size(&get_desc, sizeof(flip_param_t));
  dma_set_bsize(&get_desc, sizeof(double) * FLIP_ROW_SIZE);

  doublev4 a_v4[FLIP_CPE_ROW][FLIP_ROW_SIZE / 4], c_v4[FLIP_CPE_ROW][FLIP_ROW_SIZE / 4];
  double (*a)[FLIP_ROW_SIZE] = a_v4, (*c)[FLIP_ROW_SIZE] = c_v4;
  my_id = athread_get_id(-1);
  flip_param_t pm;
  get_reply = 0;

  dma(get_desc, gl_pm, &pm);

  while (get_reply != 1);
  asm volatile("memb");
  int a_start = pm.size / CPE_CNT * my_id;
  int a_end = pm.size / CPE_CNT * (my_id + 1);
  int row = my_id / 8;
  int col = my_id % 8;
  long t_sum = 0;
  doublev4 v4_0 = 0;

  dma_set_mask(&get_desc, 0);
  dma_set_mode(&get_desc, PE_MODE);
  dma_set_size(&get_desc, sizeof(double) * FLIP_ROW_SIZE);
  dma_set_bsize(&get_desc, sizeof(double) * FLIP_ROW_SIZE);

  dma_set_op(&put_desc, DMA_PUT);
  dma_set_size(&put_desc, sizeof(double) * FLIP_ROW_SIZE);
  dma_set_reply(&put_desc, &put_reply);
  dma_set_mode(&put_desc, PE_MODE);
  dma_set_mask(&put_desc, 0);
  dma_set_bsize(&put_desc, sizeof(double) * FLIP_ROW_SIZE);
  dma_set_stepsize(&put_desc, 0);

  int r_cnt = 1;
  get_reply = 0;
  dma(get_desc, pm.a + a_start, a);

  while (get_reply != 1);
  int i, j, k;
  /* if (row == 7){ */
  /*   for (i = 0; i < 8; i ++){ */
  /*     if (i == col){ */
  /* 	int aint = (int)a[0][0]; */
  /* 	printf("%x %x\n", (int)a[0][0], (int)a[0][FLIP_ROW_SIZE - 1]); */
  /*     } */
  /*     athread_syn(ROW_SCOPE, 0xff); */
  /*   } */
  /* } */

  for (i = 0; i < r_cnt; i ++){
    for (j = 0; j < FLIP_ROW_SIZE; j ++)
      c[i][j] = a[i][FLIP_ROW_SIZE - 1 - j];
  }
  put_reply = 0;
  dma(put_desc, pm.c + a_start, c);

  while (put_reply != 1);

  athread_syn(ARRAY_SCOPE, 0xffff);
  long t_end;
  rtc(t_end);
  t_sum += t_end - t_start;
  long f_end;
  rtc(f_end);

  if (my_id == 0)
   printf("Cycles for all parts of do_flip_slave_dma: %lld\n", t_sum);
}

