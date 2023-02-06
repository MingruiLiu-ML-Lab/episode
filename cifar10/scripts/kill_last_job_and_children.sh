ppid=$(jobs -l | tr -s " " | cut -d " " -f2) && kill ${ppid} $(ps --ppid $ppid | tr -s " " | cut -d " " -f2 | tr '\n' '\ ' | cut -d " " -f1 --complement) && fg
