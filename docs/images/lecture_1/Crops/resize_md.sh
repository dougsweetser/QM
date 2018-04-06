#!/bin/zsh
ls *jpg | perl -lane '$f = $F[0]; $g = $f; $g =~ s/.jpg/.50.jpg/; print(qq(![](images/chapter_1/$g)));'
