#include <stdlib.h>
#include <stdio.h>
#include <athread.h>
#include <fcntl.h>
#include "param.h"
#include "mpi.h"
#include "gptl.h"

extern SLAVE_FUN(do_mm_slave)(mm_param_t*);
extern SLAVE_FUN(do_mm_slave_loopo)(mm_param_t*);
extern SLAVE_FUN(do_mm_slave_vec)(mm_param_t*);
extern SLAVE_FUN(do_mm_slave_vec_loope)(mm_param_t*);

extern SLAVE_FUN(do_hop_slave)(hop_param_t*);
extern SLAVE_FUN(do_hop_slave_vec_T)(hop_param_t*);

extern SLAVE_FUN(do_accum_slave_vec)(accum_param_t*);
extern SLAVE_FUN(do_accum_slave_vec_rc)(accum_param_t*);

extern SLAVE_FUN(do_flip_slave)(flip_param_t*);
extern SLAVE_FUN(do_flip_slave_dma)(flip_param_t*);
static inline unsigned long rpcc(){
  unsigned long time;
  asm("rtc %0" : "=r" (time):);
  return time;
}

void do_mm_master(mm_param_t *pm){
  long start = rpcc();
  int i, j, k;
  for (i = 0; i < pm->size; i ++){
    for (j = 0; j < 4; j ++){
      pm->c[i][j] = 0;
      for (k = 0; k < 4; k ++)
	pm->c[i][j] += pm->a[i][k] * pm->b[k][j];
    }
  }
  printf("Cycles for do_mm_master: %lld\n", rpcc() - start);
}

void do_hop_master(hop_param_t *pm){
  long start = rpcc();
  int i;
  for (i = 0; i < pm->size; i ++){
    pm->c[i] = pm->a[i][0] / pm->a[i][1] + pm->a[i][2] / pm->a[i][3];
  }

  printf("Cycles for do_hop_master: %lld\n", rpcc() - start);
}

void do_accum_master(accum_param_t *pm){
  long start = rpcc();
  int i, j, k;

  for (k = 0; k < 4; k ++)
      pm->c[0][k] = pm->a[0][k];
  
  for (i = 1; i < pm->size; i ++)
    for (k = 0; k < 4; k ++)
      pm->c[i][k] = pm->c[i - 1][k] + pm->a[i][k];
  printf("Cycles for do_accum_master: %lld\n", rpcc() - start);
}

void do_flip_master(flip_param_t *pm){
  long start = rpcc();
  int i, j, k;
  for (i = 0; i < pm->size; i ++)
    for (j = 0; j < FLIP_ROW_SIZE; j ++)
      pm->c[i][j] = pm->a[i][FLIP_ROW_SIZE - 1 - j];
  printf("Cycles for do_flip_master: %lld\n", rpcc() - start);
}

void mm_param_init(mm_param_t *pm, int size){
  pm->size = size;
  pm->a = malloc(sizeof(double) * 4 * size);
  pm->b = malloc(sizeof(double) * 4 * 4);
  pm->c = malloc(sizeof(double) * 4 * size);
  int i, j, k;
  for (i = 0; i < size; i ++)
    for (j = 0; j < 4; j ++)
      pm->a[i][j] = i * 4 + j;
  for (i = 0; i < 4; i ++)
    for (j = 0; j < 4; j ++)
      pm->b[i][j] = i * 4 + j;
}

double mm_param_checksum(mm_param_t *pm){
  GPTLstart("mm checksum");
  double sum = 0;
  int i, j;
  for (i = 0; i < pm->size; i ++)
    sum += pm->c[i][0] + pm->c[i][1] - pm->c[i][2] - pm->c[i][3];
  GPTLstop("mm checksum");
  return sum;
}

void hop_param_init(hop_param_t *pm, int size){
  pm->size = size;

  pm->a = malloc(sizeof(double) * 4 * size);
  pm->c = malloc(sizeof(double) * size);
  int i, j;
  for (i = 0; i < size; i ++)
    for (j = 0; j < 4; j ++)
      pm->a[i][j] = i * 4 + j;
}

double hop_param_checksum(hop_param_t *pm){
  GPTLstart("hop checksum");
  double sum = 0;
  int i;
  for (i = 0; i < pm->size; i ++){
    sum += i * pm->c[i];
  }
  GPTLstop("hop checksum");
  return sum;
}

void accum_param_init(accum_param_t *pm, int size){
  pm->size = size;

  pm->a = malloc(sizeof(double) * 4 * size);
  pm->c = malloc(sizeof(double) * 4 * size);
  int i, j, k;

  for (i = 0; i < size; i ++)
      for (k = 0; k < 4; k ++)
	pm->a[i][k] = i >> k;
}

double accum_param_checksum(accum_param_t *pm){
  GPTLstart("accum checksum");
  double sum = 0;
  int i, j;
  int size = pm->size - 1;
  sum += pm->c[size][0] + pm->c[size][1] + pm->c[size][2] + pm->c[size][3];

  GPTLstop("accum checksum");
  return sum;
}

void flip_param_init(flip_param_t *pm){
  pm->size = 64;
  pm->a = malloc(sizeof(double) * pm->size * FLIP_ROW_SIZE);
  pm->c = malloc(sizeof(double) * pm->size * FLIP_ROW_SIZE);
  int i, k;
  for (i = 0; i < pm->size; i ++)
      for (k = 0; k < FLIP_ROW_SIZE; k ++)
	pm->a[i][k] = i * FLIP_ROW_SIZE + k;
}

double flip_param_checksum(flip_param_t *pm){
  GPTLstart("flip checksum");
  double sum = 0;
  int i, j;
  for (i = 0; i < pm->size; i ++){
    for (j = 0; j < FLIP_ROW_SIZE; j ++){
      sum += j * pm->c[i][j];
    }
  }
  GPTLstop("flip checksum");
  return sum;
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  GPTLinitialize();
  GPTLstart("main");
  int size = 65536;
  if (argc > 1){
    size = atoi(argv[1]);
  }
  GPTLstart("athread init");
  athread_init();
  GPTLstop("athread init");

  mm_param_t mm_pm;
  puts("Running matrix multiply examples...");

  puts("Initialize...");
  GPTLstart("mm init");
  mm_param_init(&mm_pm, size);
  GPTLstop("mm init");

  puts("Running matrix multiply on master core...");
  GPTLstart("mm master");
  do_mm_master(&mm_pm);
  GPTLstop("mm master");

  printf("Done, checksum is %f\n", mm_param_checksum(&mm_pm));

  puts("Running matrix multiply on slave cores...");
  GPTLstart("mm slave");
  athread_spawn(do_mm_slave, &mm_pm);
  athread_join();
  GPTLstop("mm slave");

  printf("Done, checksum is %f\n", mm_param_checksum(&mm_pm));

  puts("Running matrix multiply on slave cores with loop optimization...");
  GPTLstart("mm slave loopo");
  athread_spawn(do_mm_slave_loopo, &mm_pm);
  athread_join();
  GPTLstop("mm slave loopo");
  printf("Done, checksum is %f\n", mm_param_checksum(&mm_pm));

  puts("Running matrix multiply on slave cores with vectorization...");
  GPTLstart("mm slave vec");
  athread_spawn(do_mm_slave_vec, &mm_pm);
  athread_join();
  GPTLstop("mm slave vec");

  printf("Done, checksum is %f\n", mm_param_checksum(&mm_pm));

  puts("Running matrix multiply on slave cores with vectorization and loop unrolling...");
  GPTLstart("mm slave vec loope");
  athread_spawn(do_mm_slave_vec_loope, &mm_pm);
  athread_join();
  GPTLstop("mm slave vec loope");

  printf("Done, checksum is %f\n", mm_param_checksum(&mm_pm));

  puts("Running horizontal operation examples...");
  puts("Initialize");
  hop_param_t hop_pm;
  GPTLstart("hop init");
  hop_param_init(&hop_pm, size);
  GPTLstop("hop init");

  puts("Running horizontal operation on master core...");
  GPTLstart("hop master");
  do_hop_master(&hop_pm);
  printf("%f\n", hop_param_checksum(&hop_pm));
  GPTLstop("hop master");

  puts("Running horizontal operation on slave cores...");
  GPTLstart("hop slave");
  athread_spawn(do_hop_slave, &hop_pm);
  athread_join();
  printf("Done, checksum is %f\n", hop_param_checksum(&hop_pm));
  GPTLstop("hop slave");

  puts("Running horizontal operation on slave cores with transposing...");
  GPTLstart("hop slave vec T");
  athread_spawn(do_hop_slave_vec_T, &hop_pm);
  athread_join();
  printf("Done, checksum is %f\n", hop_param_checksum(&hop_pm));
  GPTLstop("hop slave vec T");

  puts("Running accumulation examples...");
  puts("Initialize");
  accum_param_t acc_pm;
  GPTLstart("accum init");
  accum_param_init(&acc_pm, size);
  GPTLstop("accum init");
  
  puts("Running accumulation on master core...");
  GPTLstart("accum master");
  do_accum_master(&acc_pm);
  printf("Done, checksum is %f\n", accum_param_checksum(&acc_pm));
  GPTLstop("accum master");

  puts("Running accumulation on slave core0 with vectorization...");
  GPTLstart("accum slave vec");
  athread_spawn(do_accum_slave_vec, &acc_pm);
  athread_join();
  printf("Done, checksum is %f\n", accum_param_checksum(&acc_pm));
  GPTLstop("accum slave vec");

  puts("Running accumulation on slave cores with vectorization and register communication");
  GPTLstart("accum slave vec rc");
  athread_spawn(do_accum_slave_vec_rc, &acc_pm);
  athread_join();
  printf("Done, checksum is %f\n", accum_param_checksum(&acc_pm));
  GPTLstop("accum slave vec rc");

  puts("Running flip examples...");
  puts("Initialize");
  flip_param_t flip_pm;
  GPTLstart("flip init");
  flip_param_init(&flip_pm);
  GPTLstop("flip init");
  
  puts("Running flip on master core...");
  GPTLstart("flip master");
  do_flip_master(&flip_pm);
  printf("Done, checksum is %f\n", flip_param_checksum(&flip_pm));
  GPTLstop("flip master");

  puts("Running flip on slave cores...");
  GPTLstart("flip slave");
  memset(flip_pm.c, 0, sizeof(double) * 64 * FLIP_ROW_SIZE);
  athread_spawn(do_flip_slave, &flip_pm);
  athread_join();
  printf("Done, checksum is %f\n", flip_param_checksum(&flip_pm));
  GPTLstop("flip slave");

  puts("Running flip on slave cores with dma...");
  GPTLstart("flip slave dma");
  memset(flip_pm.c, 0, sizeof(double) * 64 * FLIP_ROW_SIZE);
  athread_spawn(do_flip_slave_dma, &flip_pm);
  athread_join();
  printf("Done, checksum is %f\n", flip_param_checksum(&flip_pm));
  GPTLstop("flip slave dma");

  GPTLstop("main");
  GPTLpr_summary_file(MPI_COMM_WORLD, "outfile");
  return 0;
}
