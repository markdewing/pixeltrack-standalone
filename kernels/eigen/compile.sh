
EIGEN_DIR=/home/mdewing/software/linalg/eigen/eigen

clang++ \
-fopenmp \
-fopenmp-targets=nvptx64 \
"-DEIGEN_ASM_COMMENT(x)=" -DEIGEN_NO_CUDA -DEIGEN_DONT_VECTORIZE -DEIGEN_NO_CPUID \
-I ${EIGEN_DIR} \
testEigenNoFit_cpu.cc

#-DEIGEN_STACK_ALLOCATION_LIMIT=0 \
