typedef struct mm_param_t{
  double (*a)[4], (*b)[4];
  double (*c)[4];
  int size;
} mm_param_t;

typedef struct hop_param_t{
  double (*a)[4];
  double (*c);
  int size;
} hop_param_t;

typedef struct accum_param_t{
  double (*a)[4];
  double (*c)[4];
  int size;
} accum_param_t;

#define FLIP_ROW_SIZE 1024
typedef struct flip_param_t{
  double (*a)[FLIP_ROW_SIZE];
  double (*c)[FLIP_ROW_SIZE];
  int size;
} flip_param_t;
