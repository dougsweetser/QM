#!/bin/zsh
convert ../space-time_graphs_time.png -resize 9.85% small.png
ls baseball* | perl -lane '$f = $F[0]; $n = $f; $n =~ s/baseball_time/time/; print(qq(convert +append $f small.png $n));'|zsh
convert -delay 8 -loop 0 time_000* time.gif
cp time.gif ../../baseball_animations
