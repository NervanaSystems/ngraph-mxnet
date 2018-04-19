/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet

import java.io._

import org.apache.mxnet.Base._
import org.slf4j.{Logger, LoggerFactory}

/**
 * Key value store interface of MXNet for parameter synchronization.
 * @author Yizhi Liu
 */
object KVStore {

  // group id of scheduler/server/worker
  val GROUP_NODE_SCHEDULER = 1
  val GROUP_NODE_SERVER = 2
  val GROUP_NODE_WORKER = 4

  /**
   * Create a new KVStore. <br />
   * <b>
   * WARNING: it is your responsibility to clear this object through dispose().
   * </b>
   *
   * @param name : {'local', 'dist'}
   *     The type of KVStore
   *     - local works for multiple devices on a single machine (single process)
   *     - dist works for multi-machines (multiple processes)
   * @return The created KVStore
   */
  def create(name: String = "local"): KVStore = {
    val handle = new KVStoreHandleRef
    checkCall(_LIB.mxKVStoreCreate(name, handle))
    new KVStore(handle.value)
  }
}

class KVStore(private[mxnet] val handle: KVStoreHandle) extends WarnIfNotDisposed {
  private val logger: Logger = LoggerFactory.getLogger(classOf[KVStore])
  private var updaterFunc: MXKVStoreUpdater = null
  private var disposed = false
  protected def isDisposed = disposed

  /**
   * Release the native memory.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      _LIB.mxKVStoreFree(handle)
      disposed = true
    }
  }

  /**
   * Initialize a single or a sequence of key-value pairs into the store.
   * For each key, one must init it before push and pull.
   * Only worker 0's (rank == 0) data are used.
   * This function returns after data have been initialized successfully
   *
   * @param keys The keys.
   * @param values The values.
   */
  def init(keys: Array[String], values: Array[NDArray]): Unit = {
    require(keys.length == values.length, "len(keys) != len(values)")
    val valuePtrs = values.map(_.handle)
    checkCall(_LIB.mxKVStoreInitEx(handle, keys.length, keys, valuePtrs))
  }

  def init(key: String, value: NDArray): Unit = {
    init(Array(key), Array(value))
  }

  /**
   * Push a single or a sequence of key-value pairs into the store.
   * Data consistency:
   * 1. this function returns after adding an operator to the engine.
   * 2. push is always called after all previous push and pull on the same key are finished
   * 3. there is no synchronization between workers. One can use _barrier() to sync all workers
   *
   * @param keys Keys
   * @param values  According values
   * @param priority
   *         The priority of the push operation.
   *         The higher the priority, the faster this action is likely
   *         to be executed before other push actions.
   */
  def push(keys: Array[String], values: Array[NDArray], priority: Int): Unit = {
    require(keys.length == values.length, "len(keys) != len(values)")
    val valuePtrs = values.map(_.handle)
    checkCall(_LIB.mxKVStorePushEx(handle, keys.length, keys, valuePtrs, priority))
  }

  def push(keys: Array[String], values: Array[NDArray]): Unit = push(keys, values, 0)

  def push(key: String, value: NDArray, priority: Int = 0): Unit = {
    push(Array(key), Array(value), priority)
  }

  def push(key: String, values: Array[NDArray], priority: Int): Unit = {
    val keys = Array.fill(values.length)(key)
    push(keys, values, priority)
  }

  def push(key: String, values: Array[NDArray]): Unit = {
    push(key, values, 0)
  }

  /**
   * Pull a single value or a sequence of values from the store.
   *
   * Data consistency:
   * 1. this function returns after adding an operator to the engine. But any
   *    further read on out will be blocked until it is finished.
   * 2. pull is always called after all previous push and pull on the same key are finished
   * 3. It pulls the newest value from the store.
   * @param keys Keys
   * @param outs According values
   * @param priority
   *     The priority of the push operation.
   *     The higher the priority, the faster this action is likely
   *     to be executed before other push actions.
   */
  def pull(keys: Array[String], outs: Array[NDArray], priority: Int): Unit = {
    require(keys.length == outs.length, "len(keys) != len(outs)")
    val outPtrs = outs.map(_.handle)
    checkCall(_LIB.mxKVStorePullEx(handle, keys.length, keys, outPtrs, priority))
  }

  def pull(keys: Array[String], outs: Array[NDArray]): Unit = pull(keys, outs, 0)

  def pull(key: String, out: NDArray, priority: Int = 0): Unit = {
    pull(Array(key), Array(out), priority)
  }

  def pull(key: String, outs: Array[NDArray], priority: Int): Unit = {
    val keys = Array.fill(outs.length)(key)
    pull(keys, outs, priority)
  }

  def pull(key: String, outs: Array[NDArray]): Unit = {
    pull(key, outs, 0)
  }

  // Get the type of this kvstore
  def `type`: String = {
    val kvType = new RefString
    checkCall(_LIB.mxKVStoreGetType(handle, kvType))
    kvType.value
  }

  /**
   * Get the number of worker nodes
   * @return The number of worker nodes
   */
  def numWorkers: Int = {
    val size = new RefInt
    checkCall(_LIB.mxKVStoreGetGroupSize(handle, size))
    size.value
  }

  /**
   * Get the rank of this worker node
   * @return The rank of this node, which is in [0, get_num_workers())
   */
  def rank: Int = {
    val rank = new RefInt
    checkCall(_LIB.mxKVStoreGetRank(handle, rank))
    rank.value
  }

  /**
   * Register an optimizer to the store
   * If there are multiple machines, this process (should be a worker node)
   * will pack this optimizer and send it to all servers. It returns after
   * this action is done.
   *
   * @param optimizer the optimizer
   */
  def setOptimizer(optimizer: Optimizer): Unit = {
    val isWorker = new RefInt
    checkCall(_LIB.mxKVStoreIsWorkerNode(isWorker))
    if (`type`.contains("dist") && isWorker.value != 0) {
      val optSerialized = Serializer.getSerializer.serialize(optimizer)
      val cmd = Serializer.encodeBase64String(optSerialized)
      logger.debug("Send optimizer to server: {}", cmd)
      sendCommandToServers(0, cmd)
    } else {
      setUpdater(Optimizer.getUpdater(optimizer))
    }
  }

  /**
   * Set a push updater into the store.
   *
   * This function only changes the local store. Use setOptimizer for
   * multi-machines.
   *
   * @param updater  the updater function
   */
  def setUpdater(updater: MXKVStoreUpdater): Unit = {
    this.updaterFunc = updater
    checkCall(_LIB.mxKVStoreSetUpdater(handle, updaterFunc))
  }

  /**
   * Global barrier among all worker nodes
   *
   * For example, assume there are n machines, we want to let machine 0 first
   * init the values, and then pull the inited value to all machines. Before
   * pulling, we can place a barrier to guarantee that the initialization is
   * finished.
   */
  def barrier(): Unit = {
    checkCall(_LIB.mxKVStoreBarrier(handle))
  }

  def numDeadNode(nodeId: Int): Int = {
    val number = new RefInt
    checkCall(_LIB.mxKVStoreGetNumDeadNode(handle, nodeId, number))
    number.value
  }

  /**
   * Whether to do barrier when the kvstore finalizes
   * @param barrierBeforeExit
   */
  def setBarrierBeforeExit(barrierBeforeExit: Boolean): Unit = {
    val flag: Int = if (barrierBeforeExit) 1 else 0
    checkCall(_LIB.mxKVStoreSetBarrierBeforeExit(handle, flag))
  }

  /**
   * Send a command to all server nodes
   *
   * Send a command to all server nodes, which will make each server node run
   * KVStoreServer.controller
   *
   * This function returns after the command has been executed in all server nodes
   *
   * @param head the head of the command
   * @param body the body of the command
   */
  private def sendCommandToServers(head: Int, body: String): Unit = {
    checkCall(_LIB.mxKVStoreSendCommmandToServers(handle, head, body))
  }

  /**
   * Save optimizer (updater) state to file
   * @param fname Path to output states file.
   */
  def saveOptimizerStates(fname: String): Unit = {
    require(updaterFunc != null, "Cannot save states for distributed training")
    updaterFunc match {
      case cachedStates: MXKVStoreCachedStates =>
        val target = new BufferedOutputStream(new FileOutputStream(fname))
        try {
          target.write(cachedStates.serializeState())
        } finally {
          target.close()
        }
      case _ =>
        logger.warn("Updater does not have states, skip saving to {}", fname)
    }
  }

  /**
   * Load optimizer (updater) state from file
   * @param fname Path to input states file.
   */
  def loadOptimizerStates(fname: String): Unit = {
    assert(updaterFunc != null, "Cannot load states for distributed training")
    updaterFunc match {
      case cachedStates: MXKVStoreCachedStates =>
        val bis = new BufferedInputStream (new FileInputStream (fname) )
        try {
        val bArray = Stream.continually (bis.read).takeWhile (- 1 !=).map (_.toByte).toArray
          cachedStates.deserializeState(bArray)
        } finally {
          bis.close ()
        }
      case _ =>
        logger.warn("Updater does not have states, skip loading from {}", fname)
    }
  }
}
