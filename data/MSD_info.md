The entire MSD dataset lives in AWS as a public dataset snapshot.

If you have access to a server or even a PC with more than 400 gigs of (hopefully) fast storage, then you likely want to 
move the data off of AWS to your hardware. We moved the data to servers maintained by our university, since we don't 
have much funding as a student project, and we didn't want to keep on using AWS and racking up usage fees.
The whole process of moving the data off of AWS cost around 50 dollars. Most of the fee came from bandwidth fee, and not
much from compute.

Link to snapshot info: https://aws.amazon.com/datasets/million-song-dataset/

To get MSD data,
1. Make an AWS account
2. Create an EC2 instance in a specific region (I think it was us-east-1). It needs to be in the same region as the MSD snapshot is.
    - Make sure you create a key-pair for the EC2 instance, since you'll need it later.
3. Create a disk that is 500GB and attach it to your EC2 instance.
4. Attach the MSD snapshot to this 500GB disk.
5. You should be able to find the MSD somewhere in your file system, accessible from the console window in AWS.
6. Open the command line for the server you're trying to copy the data over to (We SSHed into the uni servers).
7. Create a password.pem file using the key-pair and put it somewhere on your local machine. You'll need it to ssh into the EC2 instance.
8. Use the rsync or scp command in the terminal of your local machine to start copying over the files over internet.
    - The exact command we used was: `$ rsync -aP --ignore-existing -e "ssh -i ~/private/password.pem" ec2-user@ec2-xx-xxx-xx-xx.compute-1.amazonaws.com:/mnt/snap/data ~/public`
    - the aP flag means recursively go into all directories, and be verbose (display messages in the terminal window)
    - we specify with the -e flag that we want to ssh into the ec2 machine using password.pem as the key
    - "/mnt/snap/data" is the directory where the MSD is. and "~/public" is where we want to copy the MSD to


I will provide some of the code I used to extract pertinent.

