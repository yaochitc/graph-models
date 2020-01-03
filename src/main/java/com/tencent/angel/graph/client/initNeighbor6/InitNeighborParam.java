package com.tencent.angel.graph.client.initNeighbor6;

import com.tencent.angel.PartitionKey;
import com.tencent.angel.exception.AngelException;
import com.tencent.angel.graph.utils.LongIndexComparator;
import com.tencent.angel.ml.matrix.psf.update.base.PartitionUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.base.UpdateParam;
import com.tencent.angel.psagent.PSAgentContext;
import com.tencent.angel.psagent.matrix.oplog.cache.RowUpdateSplitUtils;
import it.unimi.dsi.fastutil.ints.IntArrays;

import java.util.ArrayList;
import java.util.List;

public class InitNeighborParam extends UpdateParam {

	private long[] keys;
	private int[] indptr;
	private long[] neighbors;
	private int[] types;
	private int start;
	private int end;

	public InitNeighborParam(int matrixId, long[] keys,
							 int[] indptr, long[] neighbors) {
		this(matrixId, keys, indptr, neighbors, 0, keys.length);
	}

	public InitNeighborParam(int matrixId, long[] keys,
							 int[] indptr, long[] neighbors,
							 int start, int end) {
		this(matrixId, keys, indptr, neighbors, null, start, end);
	}

	public InitNeighborParam(int matrixId, long[] keys,
							 int[] indptr, long[] neighbors,
							 int[] types,
							 int start, int end) {
		super(matrixId);
		this.keys = keys;
		this.indptr = indptr;
		this.neighbors = neighbors;
		this.types = types;
		this.start = start;
		this.end = end;
	}

	@Override
	public List<PartitionUpdateParam> split() {
		LongIndexComparator comparator = new LongIndexComparator(keys);
		int size = end - start;
		int[] index = new int[size];
		for (int i = 0; i < size; i++)
			index[i] = i + start;
		IntArrays.quickSort(index, comparator);

		List<PartitionUpdateParam> params = new ArrayList<>();
		List<PartitionKey> parts = PSAgentContext.get().getMatrixMetaManager().getPartitions(matrixId);

		if (!RowUpdateSplitUtils.isInRange(keys, index, parts)) {
			throw new AngelException(
					"node id is not in range [" + parts.get(0).getStartCol() + ", " + parts
							.get(parts.size() - 1).getEndCol());
		}

		int nodeIndex = start;
		int partIndex = 0;
		while (nodeIndex < end || partIndex < parts.size()) {
			int length = 0;
			long endOffset = parts.get(partIndex).getEndCol();
			while (nodeIndex < end && keys[index[nodeIndex - start]] < endOffset) {
				nodeIndex++;
				length++;
			}

			if (length > 0)
				params.add(new InitNeighborPartParam(matrixId,
						parts.get(partIndex), keys, index, indptr, neighbors, types,
						nodeIndex - length - start, nodeIndex - start));

			partIndex++;
		}

		return params;
	}
}
