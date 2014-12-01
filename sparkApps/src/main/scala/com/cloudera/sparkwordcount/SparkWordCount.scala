package com.cloudera.sparkwordcount

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object SparkWordCount {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("Spark Count"))
    val threshold = args(1).toInt

    // split each document into words
    val tokenized = sc.textFile(args(0)).flatMap(_.split(" "))

    // count the occurrence of each word
    val wordCounts = tokenized.map((_, 1)).reduceByKey(_ + _)

    // Print out the values
    // wordCounts.collect.foreach(println)

    // Save the output in a file
    // wordCounts.saveAsTextFile("output.txt")

    // filter out words with less than threshold occurrences
    val filtered = wordCounts.filter(_._2 >= threshold)
    filtered.collect.foreach(println)

  }
}
