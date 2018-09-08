CUDA connected component solver. Needs pascal or volta GPU to run without modifying the makefile. 
the soc computer minion[1-17].fx.clemson.edu should work 

Included are 4 data sets, but I think only cond-mat-2003 and stokes128 meet the requirements, with 1599 and 1 connected
components, respectively. 
Usage: ./concomp path-to-file

The project just counts the connected components. Each has a unique label but they are arbitrary. Could be enhances
if a method for assigning labels was given.

bash script "autorun" will try to automatically build and run this project

individual matrix files include references

This project uses the BeBOP smc library. I do not own these utilities. http://bebop.cs.berkeley.edu/smc/ 

see further documentation in header of connected_components.cu

