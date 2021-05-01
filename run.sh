

echo 'Press Q to stop generating data'
echo
# mpirun -np 4 lmp_mpi -in ./traj/quench.txt

while true; do

    # get a random seed of 6 digits
    TEMP=$(((RND=RANDOM<<15|RANDOM)))
    TEMP=${TEMP: -5}
    TEMP2=$(((RND=RANDOM<<15|RANDOM)))
    TEMP2=${TEMP: -4}
    NAME="traj_dump-${TEMP}.ATOM"

    sed -i -E "8s/[^ ]+/$TEMP/5" ./quench.txt
    # sed -i -E "9s/[^ ]+/$TEMP2/5" ./traj/quench.txt
    sed -i -E "19s/[^ ]+/$TEMP/5" ./quench.txt
    sed -i -E "20s/[^ ]+/$TEMP/7" ./quench.txt
    sed -i -E "33s/[^ ]+/$TEMP/7" ./quench.txt
    sed -i -E "30s/[^ ]+/$NAME/6" ./quench.txt


    mpirun -np 4 lmp_mpi -in ./quench.txt

    read -t 0.25 -N 1 input
    if [[ $input = "q" ]] || [[ $input = "Q" ]]; then
        echo
        break 
    fi
done
