#!/bin/zsh
convert ../space-time_graphs_space-time.png -resize 9.85% small.png
ls baseball* | perl -lane '$f = $F[0]; $n = $f; $n =~ s/baseball_space-time/space-time/; print(qq(convert +append $f small.png $n));'|zsh
convert -delay 8 -loop 0 space-time_000* space-time.gif
