import subprocess

# list of files to execute or run
sh_list = [
    'train_PBAFN_stage1.sh', 
    'train_PBAFN_e2e.sh',
    'train_PFAFN_stage1.sh',
    'train_PFAFN_e2e.sh'
]


for sh_file in sh_list:
    print(sh_file)
    print('---------------START---------------->')
    # run the file
    process = subprocess.run(f'bash {sh_file}', shell=True)
    print('---------------END------------------>')

    if process.returncode != 0:  # if script run properly then it will reuturn 0 else it will break
        break