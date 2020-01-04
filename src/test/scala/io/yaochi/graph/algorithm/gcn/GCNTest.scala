package io.yaochi.graph.algorithm.gcn

import com.tencent.angel.conf.AngelConf
import com.tencent.angel.spark.context.PSContext
import io.yaochi.graph.dataset.CoraDataset
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{After, Before, Test}

class GCNTest {

  @Before
  def start(): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("gcn")
    conf.set(AngelConf.ANGEL_PSAGENT_UPDATE_SPLIT_ADAPTION_ENABLE, "false")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    sc.setCheckpointDir("cp")
  }

  @Test
  def testGCN(): Unit = {
    val gcn = new GCN()
    gcn.setDataFormat("dense")
    gcn.setFeatureDim(1433)
    gcn.setNumClasses(7)
    gcn.setHiddenDim(100)
    gcn.setOptimizer("adam")
    gcn.setUseBalancePartition(false)
    gcn.setBatchSize(100)
    gcn.setStepSize(0.01)
    gcn.setPSPartitionNum(10)
    gcn.setPartitionNum(1)
    gcn.setUseBalancePartition(false)
    gcn.setNumEpoch(100)
    gcn.setStorageLevel("MEMORY_ONLY")
    gcn.setTestRatio(0.5f)

    val (edges, features, labels) = CoraDataset.load("data/cora")
    val (model, psModel, graph) = gcn.initialize(edges, features, Option(labels))
    gcn.fit(model, psModel, graph)

  }

  @After
  def stop(): Unit = {
    PSContext.stop()
    SparkContext.getOrCreate().stop()
  }
}
