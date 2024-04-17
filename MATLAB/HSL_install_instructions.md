# Installation of the HSL Linear Solvers 
A license is required for the proprietary HSL Linear Solvers, but it is free to academics. You can obtain a license for the Coin-HSL package on the [HSL website](https://licences.stfc.ac.uk/product/coin-hsl). Please fill out the forms and briefly describe your project. Use your academic e-mail address when registering. The access should be granted within one working day, and you can download the source code. 

## Windows 
For Windows, the pre-compiled Windows binaries should be used. Download and extract the corresponding file (currently *CoinHSL.v2023.5.26.x86_64-w64-mingw32-libgfortran5.zip*). Save the contained folders to any location on your PC. 

Now, you need to add the path to the subfolder *bin* (e.g., *C:\CoinHSL\bin* if all subfolders of the .zip file are copied to *C:\CoinHSL*) to the *PATH* environment variable. You can do this via the *Edit the systems environment variables* GUI, which can be found via the Windows search. In the section *System Variables* (lower part), select the *PATH* variable and click on *Edit...*. Now click on *New* and type in the path to the bin folder (e.g., *C:\CoinHSL\bin*). The installation should then be complete. 

## Ubuntu 
For Ubuntu, the solvers must be built from source. Download and extract the source code (currently *coinhsl-2023.05.26.tar.gz*). Some dependencies are required for the installation. Install them by opening a terminal and executing the following command:
```bash
sudo apt-get install gfortran libblas3 libblas-dev liblapack3 liblapack-dev libmetis-dev
```
The HSL Linear Solvers must be built via the [MESON build system](https://mesonbuild.com/index.html), which can be installed via pip3: 
```bash
sudo apt-get install python3 python3-pip python3-setuptools python3-wheel ninja-build #install pip3
pip3 install meson #install MESON
```
Afterward, navigate to the folder that contains the source code and execute the following commands. Replace *YOUR_INSTALL_PATH* with the actual path where you want to install the solvers (e.g., --prefix="/home/username/HSL"). 
```bash
meson setup builddir --buildtype=release --prefix="YOUR_INSTALL_PATH"
cd builddir
meson compile
meson install
```
Afterward, you need to create a symbolic link via 
```bash
ln -s YOUR_INSTALL_PATH/lib/x86_64-linux-gnu/libcoinhsl.so YOUR_INSTALL_PATH/libhsl.so
```
 and add the install path to the *LD_LIBRARY_PATH* environment variable: 
 ```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_INSTALL_PATH
```
The solvers can now be used with IPOPT / CasADi. In order to permanently add the install path to the *LD_LIBRARY_PATH* environment variable (i.e., you do not need to execute the export command above after every login), use the command 
```bash
gedit ~/.bashrc
```
 and add the following line to your bashrc file: 
 ```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_INSTALL_PATH
```

## Using the HSL Linear Solvers with CasADi 
If the environment variable (*PATH* on Windows and *LD_LIBRARY_PATH* on Linux) contains the path to the folder where the *libhsl.dll* file is located, IPOPT will automatically recognize the solvers. If you use MATLAB on Linux, starting MATLAB via the terminal and not via an icon may be necessary to ensure the environment variable is set correctly. You can check the environment variable via the following MATLAB commands: 
```
getenv('PATH') % Windows
getenv('LD_LIBRARY_PATH') % Linux
```
To select one of the HSL linear solvers in MATLAB, use the following MATLAB commands: 
```
casadi_opts = struct();
solver_opts = struct('linear_solver', 'ma57');
opti.solver('ipopt', casadi_opts, solver_opts); % set numerical backend
```
You can select the solvers ‘ma27’, ‘ma57’, ‘ma77’, ‘ma86’ or ‘ma97’. [This page](https://licences.stfc.ac.uk/product/coin-hsl) gives a short overview of which solvers are best suited for which problem size. More options for the solver can be found in the [IPOPT documentation](https://coin-or.github.io/Ipopt/OPTIONS.html).