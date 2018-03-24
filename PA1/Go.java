import java.util.*;
class Page{
	private String url;
	private Double score;
	private int outnumber;
	private HashSet<String> inlink;
	public Page(String url){
		this.url = url;
	}
	public void setoutnumber(int outnumber){
		this.outnumber = outnumber;
	}
	public void addinlink(String link){
		inlink.add(link);
	}
	public int getoutnumber(){
		return outnumber;
	}
	public List<String> getinlink(){
		List<String> list = new ArrayList<String>(inlink);
		return list;
	}
	public double getscore(){
		return score;
	}
	public void setscore(double num){
		this.score = num;
	}

	}
public class Go{
	
	public void getgraph(String url, List<String>outlinks, Map<String, Page> map){
		Page source = new Page(url);
		source.setoutnumber(outlinks.size());
		source.setscore(1.0);
		map.put(url, source);
		for(String outlink: outlinks){
			if(!map.containsKey(outlink)){
				Page in = new Page(outlink);
				in.addinlink(url);
				map.put(outlink, in);
			}
			else{
				map.get(outlink).addinlink(url);
			}
		}
	}
	public Map<String, Page> connect(Set<String> dir){
		Map<String, Page> map = new HashMap<String, Page>();
		for(String url: dir){
			Document doc = Jsoup.connect(url).get();
			List<String> outlinks = new ArrayList<>(getOutgoingLinks(doc, url).keySet());
			getgraph(url,outlinks, map);
		}
		return map;
	}
	public Map<String, Page> calculatePR(Map<String, Page> map){
		Double d = 0.85;
		int num = 10;
		while(num --> 0){
			Map<String, Double> tmp = new HashMap();
			for(String key : map.keySet()){
				List<String> out= map.get(key).getinlink();
				double score = 1-d;
				for(String o : out){
					score += d*map.get(o).getscore()/map.get(o).getoutnumber();
				}

				tmp.put(key, score);

			}

			for(String key: tmp.keySet()){
				Double value = tmp.get(key);
				map.get(key).setscore(value);
			}

		}
		return map;
	}

	
}