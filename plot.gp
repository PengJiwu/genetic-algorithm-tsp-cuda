set term png
set output 'tour.png'
plot "tour.dat" with linespoints

set output 'opt.png'
plot "opt.dat" with linespoints
