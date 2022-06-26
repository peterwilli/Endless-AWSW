# Installing your own server

This tutorial is to host your own EAWSW server. Hosting your own server is interesting if you:

 - Are training your own model and wish to test it out in the actual game.
 - Want to run a private server and not be dependent on the public servers if they go offline or get slow during busy times.
 - Are paranoid and afraid that someone gets to know what you say to Remy when you finally get to meet him 😂

## Notes

- While it *should* be possible that this runs on Windows, I use Linux and am not sure how to run this on Windows. Feel free to adjust this tutorial for Windows, and I'll accept the PR.

## Prerequisites

You need the following software / files:

- [Docker](https://docs.docker.com/engine/install/)
- [Docker-compose](https://docs.docker.com/compose/install/)
- [Pretrained EAWSW model](https://github.com/peterwilli/Endless-AWSW/releases/tag/v0.3)

## Installation

- Download [the source from Github](https://github.com/peterwilli/Endless-AWSW/archive/refs/heads/main.zip).
- Extract the downloaded source, and open a terminal and `cd` to the `EndlessServer`-directory.
- Drop the model inside the `src/model` folder.
- To start the EAWSW server, run `docker-compose up -d`
- To test if everything is set up correctly, [click here to go to run a test command](http://localhost:5000/get_command?past=%5B%7B%22msg%22%3A+%22Hey+Remy%21%22%2C+%22cmd%22%3A+%22msg%22%2C+%22from%22%3A+%22c%22%7D%2C+%7B%22cmd%22%3A+%22scn%22%2C+%22scn%22%3A+%22park2%22%7D%2C+%7B%22msg%22%3A+%22Hey%21%22%2C+%22cmd%22%3A+%22msg%22%2C+%22from%22%3A+%22Ry%22%7D%5D&prompt=Do+I+work?).
- If you get no error dialog, you can now use `localhost:5000` as your private server in the game!