# Journal 

## [12-05-22]
### Planning
We had already clear in our minds that the ultimate goal of our project was to build a Region Detection Network but we still had to decide what do we wanna detect and where we would take all the data we needed.\
\
We decided to make the world wide famous sand box game minecraft our testbench. We figured that the cube-shaped world of the game could have made the training phase of the module faster. (and we were wrong)\

So our plan was 

### Model Selection
We then turned our attention into choosing the architechture of our model. We figured that if we wanted to succed in this project we had to discard mask based region detection since no dataset was available online and building enough ground-truth masks was too time consuming. The next choice was the natural consequence of our reasoning; Object detection based on rectangular Bounding Boxes (BBox).\
We then decided to take some time reasearching deeper into the topic and we all came to the conclusion that the Architechture we would focus was Faster-RCNN published in june 4 2015 by **Shaoqing Ren et al.** in the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497v3). 

## [13-05-22]
### The dataset
We started to scan the internet looking for any dataset that we could use, but to our surprise we did not managed to find anything appropiate. \
We then decided that we would build the dataset by ourselves from scratch. the process was not complex but indeed tedious. We recorded using OBS 1 minutes long videos of us playing in diffrent light and enviroments (desert, grass, forests, caves, ecc...) and making sure we had diffrent objects that we wanted to detect in the frame most of the times. We then coded the ["splicer"](https://github.com/Benjamin-Barda/mine-to-win/blob/main/imgs/splicer.py) that takes a video in input and generate images taken 1 frame per second.

![creeper9-00010](https://user-images.githubusercontent.com/80880329/170502171-b2a44f8b-b230-45d9-aeaa-96fa9761e92b.png)
![creeper1-00001](https://user-images.githubusercontent.com/80880329/170502252-260fa24c-e935-4ef1-a457-17987717cb84.png)
![creeper10-00052](https://user-images.githubusercontent.com/80880329/170502386-a71f9d92-946d-4e5b-90b2-4a9375e1e157.png)
![null1-00032](https://user-images.githubusercontent.com/80880329/170502656-f8623ef3-f428-4669-9e35-61f87ea7d49b.png)

While we were processing the videos we started working on the [labeling tool](https://github.com/Benjamin-Barda/mine-to-win/blob/main/utils/labelHelp.py). We needed not only to annotate what mobs where in the frames but also draw bounding boxes around them. \
This is a process that we knew was going to take a lot of effort and time but in the end we managed to get around 2000 labeled images in a weekend of work.
\
Building the dataset was a process that we continued to carry during the whole project. **For more detail about the dataset reference the document attached to this informal journal.**

## [14-05-22]
###
