Je t'ai mis les cartes ACDC de la réponse Hz/A de chaque canal dans le dossier ACDC_maps.


Ensuite, je souhaite une amélioration en vitesse du code d'optimisation matlab en sachant les composant de la machine : CPU
- Architecture: x86_64
- 2 × Intel Xeon 6760P
- 64 physical cores per socket
- 128 physical cores total
- 256 logical CPUs / threads total
- 2 threads per core
- Maximum CPU frequency: 3.8 GHz
- 2 NUMA nodes
- L2 cache: 256 MiB total
- L3 cache: 640 MiB total
- SIMD/instruction support includes:
  - AVX2
  - AVX-512
  - AVX-512 VNNI
  - AVX-512 BF16 / FP16
  - AMX
  - FMA
- 1 TiB system RAM
- Typically >600 GiB available
- NVIDIA RTX PRO 6000 Blackwell
- ~96 GB VRAM
- NVIDIA driver 580.159.03
- CUDA driver compatibility: 13.0
- GPU computing should be considered if the MATLAB operations are compatible with gpuArray.
- Local NVMe storage
- NVMe RAID available
- Fast local /scratch filesystem
- Large datasets/intermediate results should preferably use local scratch rather than network-mounted storage when I/O performance matters.
- Rocky Linux 9.8
- x86_64 Linux