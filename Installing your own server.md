# Installing your own server

This tutorial is to host your own EAWSW server. Hosting your own server is interesting if you:

 - Are training your own model and wish to test it out in the actual game.
 - Want to run a private server and not be dependent on the public servers.
 - Are paranoid and afraid that someone gets to know what you say to Remy when you finally get to meet him 😂.
 - *cough* disable the profanity / lewd filter that the public server has on by default.

## Notes

- While it *should* be possible that this runs on Windows, I use Linux and am not sure how to run this on Windows. Feel free to adjust this tutorial for Windows, and I'll accept the PR.

## Prerequisites

You need the following software / files:

- Python3 / pip
- Jina AI: `pip install jina`
- Your own model (optional, only needed if you have one)

## Installation

- Download [the source from Github](https://github.com/peterwilli/Endless-AWSW/archive/refs/heads/main.zip).
- Extract the downloaded source.
- **Optional!** If you have a pretrained model, drop the model inside the `EAWSWServer/executor/model`-folder.
    - **You can skip this step** if you wish to use ours. It'll be downloaded automatically.
- To start the EAWSW server, `cd` to the `EAWSWServer`-folder, and run `python3 serve.py`
- To test if everything is set up correctly, go to [http://localhost:5000/docs](http://localhost:5000/docs) to see the docs.
- If you get no error dialog, you can now use `localhost:5000` as your private server in the game!