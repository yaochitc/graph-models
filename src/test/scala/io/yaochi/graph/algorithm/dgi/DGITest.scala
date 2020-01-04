package io.yaochi.graph.algorithm.dgi

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.spark.context.PSContext
import io.yaochi.graph.dataset.CoraDataset
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{After, Before, Test}

class DGITest {

  @Before
  def start(): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("dgi")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("cp")
  }

  @Test
  def testDGI(): Unit = {
    val dgi = new DGI()
    dgi.setDataFormat("dense")
    dgi.setFeatureDim(1433)
    dgi.setHiddenDim(100)
    dgi.setOptimizer("adam")
    dgi.setUseBalancePartition(false)
    dgi.setBatchSize(100)
    dgi.setStepSize(0.01)
    dgi.setPSPartitionNum(10)
    dgi.setPartitionNum(1)
    dgi.setUseBalancePartition(false)
    dgi.setNumEpoch(100)
    dgi.setStorageLevel("MEMORY_ONLY")

    val (edges, features, labels) = CoraDataset.load("data/cora")
    val (model, psModel, graph) = dgi.initialize(edges, features, Option(labels))
    dgi.fit(model, psModel, graph)

  }

  @After
  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }
}
