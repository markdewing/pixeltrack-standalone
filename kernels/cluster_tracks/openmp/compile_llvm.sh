
clang++ \
-O3 \
-g \
-fopenmp \
-fopenmp-targets=nvptx64 \
-D GPU_DEBUG \
cluster_tracks_by_density.cpp

