nx=$1
ny=$2
nt=$3

# Create directories for temporary output
mkdir ./output$nx$ny$nt
mkdir ./figures$nx$ny$nt

# Clean up
rm ./output$nx$ny$nt/*.raw
rm ./figures$nx$ny$nt/*.png

# Run executable
echo './hw_4 '$nx' '$ny
OMP_PROC_BIND=true ./hw_4 $nx $ny $nt

# Create movie (install ffmpeg from https://www.ffmpeg.org/)
python3 plot.py  --path=./output$nx$ny$nt --nx=$nx --ny=$ny --output=./figures$nx$ny$nt --batch --gpu

#rm ./movie.mp4; ffmpeg -f image2 -pattern_type glob -i './figures/*.png' -c:v libx264 -vf fps=25 -pix_fmt yuv420p movie.mp4
