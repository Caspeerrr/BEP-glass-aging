

echo 'Press Q to stop generating data'
echo
# mpirun -np 4 lmp_mpi -in ./traj/quench.txt

for filename in ../dump/young/*.YOUNG; do

    TEMP="$(cut -d'-' -f2 <<<$filename)"
    TEMP2="$(cut -d'-' -f3 <<<$filename)"
    TEMP2="$(cut -d'.' -f1 <<<$TEMP2)"
    NAME="traj_dump-${TEMP}-${TEMP2}.YOUNG"

    sed -i -E "8s/[^ ]+/$TEMP/5" ../input-script/quench_young.txt
    sed -i -E "9s/[^ ]+/$TEMP2/5" ../input-script/quench_young.txt
    sed -i -E "19s/[^ ]+/$TEMP/5" ../input-script/quench_young.txt
    sed -i -E "20s/[^ ]+/$TEMP/7" ../input-script/quench_young.txt
    sed -i -E "34s/[^ ]+/$TEMP/7" ../input-script/quench_young.txt
    sed -i -E "31s/[^ ]+/$NAME/6" ../input-script/quench_young.txt

    ( cd ../dump/young ; mpirun -np 4 lmp_mpi -in ../../input-script/quench_young.txt )

done
