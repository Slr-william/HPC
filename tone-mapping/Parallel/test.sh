#!/bin/bash
#
#SBATCH --job-name=test_tone_mapping
#SBATCH --output=res_tone_mapping.md
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

echo "imagen|tiempo"
echo ":---:|:---:"

for i in {1..20}
do
	srun ./test ../images/test$i.exr 0.4 1.2 0 results/test$i.png
	#echo "**Image**: img$i.jpg"
	#echo
	#echo "iteracion|Host|OpenCV|aceleracion Host-OCV|OpenCVGPU|aceleracion OCV-OCVGPU|Cuda|aceleracion OCV-Cuda|aceleracion OCVGPU-Cuda"
	#echo ":---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:"
	#for run in {1..20}
	#do
		#argv=$((run*100))
		#echo -n "$run|"
		#./test images/test$i.exr 0.4 1.2 0 results/test$i.png
	#done
	#echo "Promedios:| | | | | | | | "
	#echo
done
