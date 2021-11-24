#!/bin/sh

cd ~/Downloads
cd h2o-3.34.0.4
java -cp h2o.jar hex.genmodel.tools.PrintMojo --tree 0 -i "/Users/gabrieltaylor/Python/STAT527/model.zip" -o model.gv -f 20 -d 3
dot -Tpng model.gv -o /Users/gabrieltaylor/Python/STAT527/imgs/model.png
open model.png