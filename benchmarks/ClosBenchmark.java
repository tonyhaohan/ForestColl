// MCNF ClosD adapter. ForestColl's three algorithm classes remain unmodified.
import java.nio.file.*;
import java.io.*;
import java.util.*;
import org.jgrapht.graph.*;
import org.jgrapht.alg.util.Pair;

public class ClosBenchmark {
    public static void main(String[] args) throws Exception {
        var graph = new DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>(DefaultWeightedEdge.class);
        Set<String> compute = new LinkedHashSet<>();
        for (String line : Files.readAllLines(Path.of(args[0]))) {
            String[] p = line.split(" ");
            if (p[0].equals("V")) {
                graph.addVertex(p[1]);
                compute.add(p[1]);
            } else {
                graph.addVertex(p[1]);
                graph.addVertex(p[2]);
                graph.addEdge(p[1], p[2]);
                graph.setEdgeWeight(p[1], p[2], Integer.parseInt(p[3]));
            }
        }
        try (var out = new PrintWriter(args[1])) {
            long t = System.nanoTime();
            var uk = new BinarySearchParallel(graph, compute).search();
            double search = (System.nanoTime() - t) / 1e9;
            out.printf(Locale.ROOT, "P %.17g %d %.9f%n", uk.getFirst(), uk.getSecond(), search);
            out.flush();
            t = System.nanoTime();
            var split = new EdgeSplittingParallel(graph, compute, uk.getSecond(), uk.getFirst()).removeSwitchNodes();
            out.printf(Locale.ROOT, "S %.9f%n", (System.nanoTime() - t) / 1e9);
            out.flush();
            for (var entry : split.getSecond().entrySet()) {
                for (var middle : entry.getValue().entrySet()) {
                    out.printf("E %s %s %s %d%n", entry.getKey().getFirst(), entry.getKey().getSecond(),
                               middle.getKey(), middle.getValue());
                }
            }
            t = System.nanoTime();
            var forest = new BranchingGenerator(split.getFirst(), uk.getSecond()).generate();
            out.printf(Locale.ROOT, "B %.9f%n", (System.nanoTime() - t) / 1e9);
            var keys = new ArrayList<>(forest.getFirst().keySet());
            keys.sort(Comparator.<Pair<String, Integer>, String>comparing(Pair::getFirst).thenComparingInt(Pair::getSecond));
            for (var key : keys) {
                out.printf("T %s %d %d", key.getFirst(), key.getSecond(), forest.getSecond().get(key));
                for (var edge : forest.getFirst().get(key)) {
                    out.printf(" %s %s", edge.getFirst(), edge.getSecond());
                }
                out.println();
            }
        }
    }
}
