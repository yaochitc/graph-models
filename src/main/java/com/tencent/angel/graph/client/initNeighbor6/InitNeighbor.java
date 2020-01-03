package com.tencent.angel.graph.client.initNeighbor6;

import com.tencent.angel.graph.data.Node;
import com.tencent.angel.ml.matrix.psf.update.base.PartitionUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.base.UpdateFunc;
import com.tencent.angel.ps.storage.vector.ServerLongAnyRow;

/* Mini-batch version of pushing csr neighbors */
public class InitNeighbor extends UpdateFunc {

	public InitNeighbor(InitNeighborParam param) {
		super(param);
	}

	public InitNeighbor() {
		this(null);
	}

	@Override
	public void partitionUpdate(PartitionUpdateParam partParam) {
		InitNeighborPartParam param = (InitNeighborPartParam) partParam;
		ServerLongAnyRow row = (ServerLongAnyRow) (psContext.getMatrixStorageManager().getRow(param.getPartKey(), 0));

		long[] keys = param.getKeys();
		long[][] neighborArrays = param.getNeighborArrays();
		int[][] neighborTypes = param.getTypeArrays();

		row.startWrite();
		try {
			for (int i = 0; i < keys.length; i++) {
				Node node = (Node) row.get(keys[i]);
				if (node == null) {
					node = new Node();
					row.set(keys[i], node);
				}

				node.setNeighbors(neighborArrays[i]);
				if (neighborTypes != null)
					node.setTypes(neighborTypes[i]);
			}
		} finally {
			row.endWrite();
		}

	}

}
