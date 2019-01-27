---
title: Is garbage in always garbage out?
date: Aug 6th 18
draft: false
author: Andrew Escay
authorImage: img/headshot.jpeg
image: /img/dssg_blog.png
comments: true
share: true
type: post
---
People who have been interested in any form of data science work have often heard these two sayings:

1. you will spend more time cleaning your data than running analysis on it; and  

2. garbage in, garbage out.  

In any data science project, clean data is almost *always* a prerequisite for accurate outcomes. There are several reasons for this:  

1. our analysis can impact real lives on the ground if we’re working with community-facing information and we don’t want to make biased conclusions;  

2. many complex analyses hold certain assumptions that must be true in order for the conclusions to be true as well; and more generally  

3. you can’t draw conclusions from data that does not accurately reflect what you are analyzing.

More specifically, in *supervised* machine learning, “ground truth” – data which is deemed to be objectively true – is the prerequisite before running any complex algorithms. This is because you are essentially telling a machine what is right and wrong, and asking it to figure out how to reach those conclusions on its own. It’s this type of data cleaning which I will focus on here. The current practice in supervised machine learning is to have people manually go through the data set that will be used to train an algorithm and filter out any messiness in the data to create what is known as “clean data” or “ground truth”.

Luckily for data scientists, many companies, industries and organizations are starting to recognize the exponentially increasing impact that data has on our lives, and are thus starting to make the necessary adjustments to support modern data infrastructures that will improve the cleanliness of data. However, this change is met with some challenges. First, many small and medium-size enterprises and organizations who wish to adopt modern data practices are constrained. They lack the funds necessary to support that infrastructure and the number of skilled professionals necessary to undertake such a massive data migration task. Second,there is still a huge amount of data that remains messy no matter how much effort you put in to try to organize it. Examples of this are: written feedback sent to companies by their customers which have no common structure and could be muddled with language errors; inventory systems that are automated but still require manual inputs, exposing them to human error; and satellite imagery that can never be devoid of some amount of cloud cover, which may obscure key details and reduce the accuracy of images.

### Why isn’t this mess going away?

As you can imagine, these types of messy data can still be generated despite your best efforts in trying to keep your database clean. They also happen to be some of the hardest data to validate, detect, and effectively clean in an automated manner – because that data would definitely not be there if it were easy to clean. The problem of satellite imagery with messy cloud cover is what I worked on over the summer at the University of Washington eScience Institute [Data Science for Social Good program](http://escience.washington.edu/dssg/). Our [Disaster Damage Detection team](http://escience.washington.edu/2018-data-science-for-social-good-projects/) is working with this imagery to build an algorithm that can annotate damages after a hurricane to help inform first responders and emergency personnel on the ground. Cloudy images are something you always have to deal with if you’re trying to generate timely images after a storm hits. It’s also very unlikely that all countries have access to expensive satellite imagery with 8 remote sensing bands, which in some cases can peer through the clouds and provide visibility. To put it bluntly, the mess is inevitable.

![Figure A](http://escience.washington.edu/wp-content/uploads/2018/08/Andrew-Escay-image-1.png)

**Figure A:** *The red boxes in the left image show buildings in the area using a data set of building footprints, but the satellite cannot detect them because it lacks the visibility to see beneath the clouds.*

The figure above shows us an example of this inevitable mess. 

### Thought Experiment - Can we perfectly clean all our  data?

In such messy data context, I pose the question: do all forms of data actually have the ability to become “clean”? Join me as we engage in a quick thought experiment . 

Let’s think of a way to try cleaning a data set with 50,000 images and a couple hundred building structures inside each image. This is not exactly how our data set was, but some features are shared. Imagine each image has a mix of cloud cover and volunteer-sourced points that indicate if a building is damaged. 

There are a few things to note:  

1. some buildings are incorrectly tagged when volunteers sort the images  

2. some images have weirdly angled shots that make buildings look somewhat distorted  

3. it is generally hard to see if there’s actual damage in the image. 
   
Now let’s try to clean it!

In the context of a few hundred points, maybe even a couple of thousand, manually inspecting and filtering out parts of the image covered by clouds can be reasonable, though difficult. However, when you start to talk about tens of thousands of points, it can get cumbersome to manually go through it all. But for the sake of this thought experiment let’s say we can do it. 

Even in the most ideal situation, where you can get a team to clean all those points, the chances that you are able to replicate this process are slim, because not everyone can source that many volunteers to filter the data. But again, let’s assume that constraint doesn’t exist. 

Even if you can get enough people to look at all of the images, the likelihood for human error comes back into play. It seems as though you will never be able to clean that data. Now let’s say you can clean it without human error, how does your model react with messy data? Are you going to start overfitting your model to work only with your data set or another perfectly cleaned data set? This is likely to be the case in a supervised machine learning process because the machine can only learn what you are going to train it with.

One method for confronting this issue is “data augmentation” which is the act of adding slight variations to the data you use to train your model in order to make it more robust, such as changing the size, color, or orientation. However, this injects some level of variation - or framed differently, mess - back into the data set. Although this statement may draw criticism, I know there is a difference between the variation you generate with augmentation and the mess of bad data. Variability in data augmentation keeps features relatively the same, you just want the computer to account for some difference in how that correct data is presented, whereas messy data is wrong information that muddles up the good information. However, there are examples where data augmentation creates more problems for the algorithm, such as when it is biased to your choice of transformations which may not be realistic.  Or on the flip side, the augmentation you generate might not be that different from data that are considered to be “messy”.

I do know that not all data problems have this inevitable mess problem, but this is where I want to draw some attention: don’t we want to be able to build algorithms robust enough to deal with a wide array of variation in our data, regardless of how it’s formed? 

### What do we do with our *landfills* of data?

The amount of data we generate year after year is exponentially increasing; we hear and see this all the time when we see posts that compare the data generated on the internet now to those of the times of Shakespeare and the like. And the majority of the data we’re generating nowadays is also *user-generated data*; posts, messages, pictures, all with their own faults, misspellings, and quirks. Thus it seems like we are essentially creating landfills of data according to our current standards of clean data. This is why I want to challenge myself and others to think of ways we can build data science algorithms and tools which can be robust to the mess we see in our data today. 

I do believe that cleaning should still be done in the long run, but I think there is a point in time when it takes too much effort to clean data for a very marginal amount of gain. I’d like to demonstrate this with a simple example visualized below. The image on the left displays the type of “messy” data we used to train the algorithm. The blue boxes here represent a damaged/flooded building, and the orange boxes represent buildings that were not damaged. The data we used to train this algorithm on was not entirely correct in flagging what was flooded or not. However, despite having some wrong labels, the computer was still able to make general conclusions about which buildings were flooded in the images on the right.

![Figure B](https://escience.washington.edu/wp-content/uploads/2018/08/Andrew-Escay-image-2-768x377.png)
**Figure B:** *The image on the left displays the type of “messy” data we used to train the algorithm. The blue boxes here represent a damaged/flooded building, and the orange boxes represent buildings that were not damaged.*

This is an isolated example where we saw this behavior but we do not know yet whether the algorithm performs this way with all other cases. However, it does give us some hope that maybe it is possible to have models that can deal with this sort of messy data.

I’ll close this off with two points which we can continue to ponder on, and hopefully learn the answers to:

1. can we build algorithms that can sift through some level of messy data and still learn effectively?; and  

2. can we make it more acceptable to work with a degree of messy data, under the assumption that you flag the bias in your model, and be very explicit about what impact this may have?

----
This is a repost of the original blog post I wrote for the eScience Institute's 2018 Data Science for Social Good Program. You can find this original piece posted [here](https://escience.washington.edu/social-good-summer-blog-issue-six/).
