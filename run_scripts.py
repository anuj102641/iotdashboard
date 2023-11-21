import subprocess

def run_script(script_path):
    subprocess.Popen(['python', script_path])

if __name__ == "__main__":
    # Replace 'script1.py', 'script2.py', and 'script3.py' with the paths to your Python scripts
    script1_path = 'main.py'
    script2_path = 'main2.py'
 #   script3_path = 'script3.py'

    # Run each script in a separate process
    run_script(script1_path)
    run_script(script2_path)
 #   run_script(script3_path)
