paste $1 $2 | grep LogProb: | awk 'BEGIN{cum=0;dif=0}{if($1=="RefMap:"){dif=$4-$10;cum+=dif;printf"MapNum: %6d Mod1: %10.5f Mod2: %10.5f Dif: %10.5f Cum: %10.5f\n",$2,$4,$10,dif,cum}}'
