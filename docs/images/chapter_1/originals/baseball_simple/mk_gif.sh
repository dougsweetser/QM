#!/bin/zsh
convert ../space-time_graphs_simple.png -resize 9.85% small.png
ls baseball* | perl -lane '$f = $F[0]; $n = $f; $n =~ s/baseball_simple/simple/; print(qq(convert +append $f small.png $n));'|zsh
convert -delay 8 -loop 0 simple_000* simple.gif
cp simple.gif ../../baseball_animations
