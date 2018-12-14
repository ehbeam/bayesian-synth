#!/usr/bin/python

import os, shutil

names = ["llda_fulltexts", "llda_abstracts"]

for name in names:

    # shutil.copyfile("wrap.sh", "{}/wrap.sh".format(name))
    shutil.copyfile("llda.py", "{}/llda.py".format(name))
    # os.makedirs("{}/logs".format(name))

    for i in range(500):
        
        comm = "llda.run_llda('{}', {})".format(name, i)
        pyfile = open("{}/llda_{}.py".format(name, i), "w+")
        pyfile.write("#!/bin/python\n\nimport llda\n{}".format(comm))
        pyfile.close()
        bashfile = open("{}/llda_{}.sbatch".format(name, i), "w+")
        time = "00-12:00:00"
        lines = ["#!/bin/bash\n",
                 "#SBATCH --job-name={}_{}".format(name.split("_")[1][:3], i),
                 "#SBATCH --output=logs/{}_{}.%j.out".format(name, i),
                 "#SBATCH --error=logs/{}_{}.%j.err".format(name, i),
                 "#SBATCH --time={}".format(time),
                 "#SBATCH -p normal",
                 "#SBATCH --mail-type=FAIL",
                 "#SBATCH --mail-user=ebeam@stanford.edu\n",
                 "module load python/2.7.13",
                 "srun python llda_{}.py".format(i)]
        for line in lines:
            bashfile.write(line + "\n")
        bashfile.close()