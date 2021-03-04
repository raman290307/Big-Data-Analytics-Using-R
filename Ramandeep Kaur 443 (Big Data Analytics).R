# My Project is on using datasubset from social media and converting it into word clouds using programmed R script
#This is created in R studio on windows desktop By : Ramandeep Kaur for BIg data analytics, Roll no 443.



#Cleaning the workspace Before begining any processing
rm(list=ls())

# Step 1 : Installing the caret (Classification and REgression TRaining package, used to build predictive models)
#install.packages(c("caret","tm","wordcloud"))

# Step 2 : Using the installed caret package, and text mining package (tm), which comes with R
library(caret)
library(tm)
library(wordcloud)

c("caret","tm","wordcloud")

# Step 3 : Setting the working directory path
dir <- "/Users/naironics/Downloads/NFL_Data_Analysis/"
setwd(dir)

# Step 4 : Printing the current working directory to console 
wd <- getwd()
cat("Current working dir: is", wd)

# Step 5 : Reading NFL dataset to 
nfldata <- read.csv("NFL_SocialMedia_sample_data.csv")


# head(data)


# Step 6: Converting the machine log into Text Corpus
textcorpus <- Corpus(VectorSource(nfldata$content))

#inspect(msg_corpus[1:5])

# Steps 7 - 11 : Data Preprocessing steps 

# Step 7 : Data Cleaning to convert textcorpus content to lower case
cleanedcorpus <- tm_map(textcorpus, content_transformer(tolower))

# Step 8 : Data Cleaning to remove the stop words from textcorpus
cleanedcorpus <- tm_map(cleanedcorpus, removeWords, stopwords())

# Step 9 : Data Cleaning to remove the punctuations from textcorpus
cleanedcorpus <- tm_map(cleanedcorpus, removePunctuation)

# Step 10 : Data Cleaning to remove the numbers from textcorpus
cleanedcorpus <- tm_map(cleanedcorpus, removeNumbers)

# Step 11 : Data Cleaning to eliminate the white spaces from textcorpus
cleanedcorpus <- tm_map(cleanedcorpus, stripWhitespace)
inspect(cleanedcorpus[1:5])

# Step 12 : Creating a Document Term Matrix
doc_term_matrix <- DocumentTermMatrix(cleanedcorpus)
inspect(doc_term_matrix[1:10,1001:1010])

# Step 13 : Determining the Term Frequency and tf_idf
dtm_tfxidf <- weightTfIdf(doc_term_matrix)
inspect(dtm_tfxidf[1:10,1001:1010])

# Step 14 : Generating matrix from dtm_tfxidf
mat <- as.matrix(dtm_tfxidf)
rownames(mat) <- 1:nrow(mat)

# Step 15 : Normalizing the Vectors , so that Euclidean makes sense
normalizeEuclidean <- function(mat) mat/apply(mat, MARGIN=1, FUN=function(x) sum(x^2)^.5)
matrixNormalized <- normalizeEuclidean(mat)

#Step 16 : K-means Clustering of matrix into 10 clusters
kmeansCluster <- kmeans(matrixNormalized, 10)
table(kmeansCluster$cluster)

# Output the clusters into Cluster_Out.csv file
x <- data.frame(Log = nfldata$content, Cluster = kmeansCluster$cluster)
write.csv(x, file = "/Users/naironics/Downloads/NFL_Data_Analysis/Cluster_Out.csv", row.names=TRUE)

# Reading back the saved output clusters file
cluster_data <- read.csv("/Users/naironics/Downloads/NFL_Data_Analysis/Cluster_Out.csv")

# Creating a Sparse Matrix out of Document Term Matrix
sparse_dtm <- removeSparseTerms(doc_term_matrix, sparse= 0.9999)
new_dtm <- as.matrix(sparse_dtm)

for (N in 1:length(kmeansCluster$withinss)) {
  a <- sort(colSums(new_dtm[kmeansCluster$cluster == N, ]),
            decreasing = TRUE)
  df <- data.frame(names(a), a)
  colnames(df) <- c("word","count")
  
  if (N == 1){
    x <- data.frame(N, length(which(cluster_data$Cluster == N )), df$word[1:5], 
                    df$count[1:5], as.numeric(rownames(x))[1:5])
    colnames(x) = c("Loggroup", "Logcount", "Top Words", "Word Count", "Counter")
  } else {
    y <- data.frame(N, length(which(cluster_data$Cluster == N )), df$word[1:5],
                    df$count[1:5], as.numeric(rownames(x))[1:5]) 
    colnames(y) = c("Loggroup", "Logcount", "Top Words", "Word Count", "Counter")
    x <- rbind(x, y)
  }
}

# Write Topwords into Topwords CSV file
write.csv(x, file = "/Users/naironics/Downloads/NFL_Data_Analysis/Topwords.csv", row.names=FALSE)


# Print Word Cloud of each cluster
for (n in 1:10){
  print(c("Cluster", n), quote=F)
  color <- brewer.pal(8,"Dark2")
  wordcloud(cleanedcorpus[kmeansCluster$cluster==n],  min.freq=5, 
            max.words=30, colors=color, random.order=FALSE)
}

