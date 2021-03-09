# rambles
A live per-process RAM usage pie chart in python using
[matplotlib](https://matplotlib.org/) and
[ps_mem](https://github.com/pixelb/ps_mem).

# Install
`git clone git@github.com:astromancer/ramble.git`\
`alias ramble='python $PWD/ramble'`\
Add the line above to your bashrc or bash_aliases if you want it to always be available


# Usage
Launch the main event loop. \
`ramble` 


![Example image](https://github.com/astromancer/ramble/blob/master/example.png?raw=True "Per-process RAM usage")


Here's an example of me launching firefox. The ffmpeg usage also creeps up as
as this animation is being built.

![Example gif](https://github.com/astromancer/ramble/blob/master/example.gif?raw=True "Per-process RAM usage")
