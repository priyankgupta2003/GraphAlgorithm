import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Job;

class NodePageRank implements WritableComparable<NodePageRank>{

	IntWritable  nodeId;
	DoubleWritable pagerank;
	
	public NodePageRank(){
		set(new IntWritable(), new DoubleWritable());
	}
	
	void set (NodePageRank np){
		nodeId = np.nodeId;
		pagerank = np.pagerank;
	}
	
	void set (IntWritable n, DoubleWritable p){
		nodeId = n;
		pagerank = p;
	}
	
	public NodePageRank(IntWritable nodeId, DoubleWritable pagerank) {
		super();
		this.nodeId = nodeId;
		this.pagerank = pagerank;
	}

	public IntWritable getNodeId() {
		return nodeId;
	}

	public void setNodeId(IntWritable nodeId) {
		this.nodeId = nodeId;
	}

	public DoubleWritable getPagerank() {
		return pagerank;
	}

	public void setPagerank(DoubleWritable pagerank) {
		this.pagerank = pagerank;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		nodeId.readFields(in);
		pagerank.readFields(in);
		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		nodeId.write(out);
		nodeId.write(out);
	}

	@Override
	public int compareTo(NodePageRank n) {
		int cmp = nodeId.compareTo(n.nodeId);
		return cmp;
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof NodePageRank){
			NodePageRank n = (NodePageRank) o;
			return nodeId.equals(n.nodeId);
		}
		return false;
	}
	
}

class Node implements WritableComparable<Node>{
	
	IntWritable nodeId;
	IntWritable links;
	
	Node(){
		set(new IntWritable(), new IntWritable());
	}
	
	Node( Node n){
		nodeId = n.nodeId;
		links = n.links;
	}
	
	Node (Integer n, Integer l){
		set(new IntWritable(n), new IntWritable(l) );
	}
	
	Node (IntWritable n, IntWritable l){
		nodeId = n;
		links =l;
	}
	
	void set (IntWritable n, IntWritable l){
		nodeId = n;
		links = l;
	}
	
	void set( Node n){
		nodeId = n.nodeId;
		links = n.links;
	}
	
	public IntWritable getNodeId(){
		return nodeId;
	}
	
	public IntWritable getLinks(){
		return links;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		nodeId.readFields(in);
		links.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		nodeId.write(out);
		links.write(out);
		
	}
	
	@Override
	public boolean equals(Object o){
		if(o instanceof Node){
			Node n = (Node) o;
			return nodeId.equals(n.nodeId);
		}
		return false;
	}
	
	
	@Override
	public int compareTo(Node n) {
		int cmp = nodeId.compareTo(n.nodeId);
		return cmp;
	}
	
	public String toString(){
		return ("nodeId: "+nodeId.toString() + "\t" + "number of links: "+links.toString());
	}

}


public class PageRank {
	
	public static class PreGraphMapper extends Mapper<Object, Text, IntWritable,IntWritable>{
		static IntWritable src = new IntWritable();
		static IntWritable dest = new IntWritable();
		
		public void map(Object k, Text v, Context context) throws IOException, InterruptedException{
			try{
				String[] s = v.toString().split(",");
				src = new IntWritable(Integer.parseInt(s[0]));
				dest = new IntWritable(Integer.parseInt(s[1]));
				context.write(src, dest);
			}catch(Exception e){
				System.out.println("****Wrong Input Line Found****");
			}
		}
	}
	
	public static class PreGraphReducer extends Reducer<IntWritable, IntWritable, Node, IntWritable>{
		static Node src = new Node();
		static IntWritable dest = new IntWritable();
		
		public void reduce(IntWritable k, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
			int count = 0;
			//System.out.println("PreGraphReducer Key (from) = "+k.toString());
			ArrayList<Integer> vals = new ArrayList<Integer>();
			for(IntWritable val:values){
				count++;
				Integer copy = val.get();
				vals.add(copy);
			}
			for(int i=0;i<vals.size();i++){
				src.set(k, new IntWritable(count));
				dest = new IntWritable(vals.get(i));
				//System.out.println(to.toString()+" = "+from.toString());
				context.write(src,dest);
			}
		}
	}
	
	public static class FirstIterationMapper extends Mapper <Node, IntWritable, IntWritable, DoubleWritable> {
		
		static IntWritable n = new IntWritable();
		static DoubleWritable p = new DoubleWritable();
		static double initialPageRank = 0.8;
		
		public void map(Node src, IntWritable dest, Context context) throws IOException, InterruptedException{
			
			Double d = initialPageRank/src.getLinks().get();
			n = new IntWritable(dest.get());
			p = new DoubleWritable(d);
			
			context.write(n, p);		
		}
		
	}
	
	public static class FirstIterationReducer extends Reducer <IntWritable, DoubleWritable, IntWritable, DoubleWritable> {
		
		static double damping_factor = 0.5;
		static int no_nodes = 100;
		
		public void reduce(IntWritable n, Iterable<DoubleWritable> p, Context context) throws IOException, InterruptedException{
			
			double pagerank_sum = 1-damping_factor/no_nodes;
			for(DoubleWritable val : p){
				pagerank_sum = pagerank_sum + (damping_factor*val.get());
			}
			
			DoubleWritable pr = new DoubleWritable(pagerank_sum);		
			context.write(n, pr);
	
		}
		
	}
	
	public static class IterativeMapper extends Mapper <Node, IntWritable, IntWritable, DoubleWritable>{
		
		static IntWritable n = new IntWritable();
		static DoubleWritable p = new DoubleWritable();
		
		private static HashMap<Integer, Double> pagemap = new HashMap<Integer, Double>();
		private BufferedReader brReader;
		
		protected void setup(Context context) throws IOException{
			
			String str = "";
			
			URI[] cacheLocalFiles = context.getCacheFiles();
			for (URI u : cacheLocalFiles){
				Path path = new Path(u.toString());
				try{
					brReader = new BufferedReader(new FileReader(path.toString()));
					while ((str = brReader.readLine()) != null) {
						String pageRankArray[] = str.split("\\t");
						pagemap.put(Integer.parseInt(pageRankArray[0].trim()),Double.parseDouble(pageRankArray[1].trim()));
					}
				}catch(Exception e){
					System.out.println("File Read not Successful");
					e.printStackTrace();
				}finally {
					if (brReader != null) {
						brReader.close();
					}
				}
			}		
		}
		
		public void map(Node src, IntWritable dest, Context context) throws IOException, InterruptedException{
			
			Integer srcNode = src.getNodeId().get();
			Double srcRank = pagemap.get(srcNode);
			Double d = srcRank/src.getLinks().get();
			n = new IntWritable(dest.get());
			p = new DoubleWritable(d);
			
			context.write(n, p);
			
		}
		
		public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException, URISyntaxException {
			
			Configuration conf = new Configuration();
			Job job = Job.getInstance(conf, "PreGraphProcessing");
			job.setJarByClass(PageRank.class);
			
			job.setMapperClass(PreGraphMapper.class);
			job.setReducerClass(PreGraphReducer.class);
			job.setOutputKeyClass(Node.class);
			job.setOutputValueClass(IntWritable.class);
			FileInputFormat.addInputPath(job, new Path("graph")); 
		    FileOutputFormat.setOutputPath(job, new Path("output1"));
		    
		    job.waitForCompletion(true);
			
			
			conf = new Configuration();
			job = Job.getInstance(conf, "First Iteration");
			job.setJarByClass(PageRank.class);
			
			job.setMapperClass(FirstIterationMapper.class);
			job.setReducerClass(FirstIterationReducer.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(DoubleWritable.class);
			
			FileInputFormat.addInputPath(job, new Path("output1")); 
		    FileOutputFormat.setOutputPath(job, new Path("pagerank0"));
		    
		    job.waitForCompletion(true);
		    
		    conf = new Configuration();
			job = Job.getInstance(conf, "Iterations");
			job.setJarByClass(PageRank.class);
			
			job.setMapperClass(IterativeMapper.class);
			job.setReducerClass(FirstIterationReducer.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(DoubleWritable.class);
			
			FileSystem fs = FileSystem.get(new URI("hdfs://localhost:9000/pagerank0"), conf);
			FileStatus[] fileStatus = fs.listStatus(new Path("hdfs://localhost:9000/pagerank0"));
			for(FileStatus status : fileStatus){
		        job.addCacheFile(status.getPath().toUri());
		    }
			
			FileInputFormat.addInputPath(job, new Path("output1")); 
		    FileOutputFormat.setOutputPath(job, new Path("pagerank0"));
			
			
		}
		
		
	}
	
}
