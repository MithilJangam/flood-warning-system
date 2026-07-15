import subprocess
import webbrowser
import time
import sys
import os

def main():
    use_shell = os.name == 'nt'

    # Auto-install Node.js dependencies if node_modules is missing
    if not os.path.exists("node_modules"):
        print("node_modules folder not found. Automatically installing dependencies...")
        try:
            subprocess.run(["npm", "install"], shell=use_shell, check=True)
            print("✓ Dependencies installed successfully!\n")
        except FileNotFoundError:
            print("Error: 'npm' command not found. Please ensure Node.js is installed on your PC.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error during npm install: {e}")
            sys.exit(1)

    print("Starting Flood Warning System fullstack server...")
    
    try:
        # Start node process
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except FileNotFoundError:
        print("Error: 'npm' command not found. Please ensure Node.js is installed.")
        sys.exit(1)

    print("Waiting for server to bind to port 3000...")
    time.sleep(3.0)
    
    print("Opening browser at http://localhost:3000...")
    webbrowser.open("http://localhost:3000")
    
    print("Server logs (Press Ctrl+C to terminate):")
    try:
        # Stream logs
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
    except KeyboardInterrupt:
        print("\nTerminating server...")
        if os.name == 'nt':
            # On Windows, kill the entire process tree to prevent orphaned node processes
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    main()
