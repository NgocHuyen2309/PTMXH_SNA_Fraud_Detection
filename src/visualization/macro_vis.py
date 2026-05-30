
import os
import sys
import argparse
import logging
from pathlib import Path
import networkx as nx

# Configure logging for academic reporting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


class MacroVisualizer:
    """Class to handle macro-level topological analysis and export for Gephi visualization."""

    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the visualizer with input and output paths.
        
        :param input_path: Path to the input GraphML file.
        :param output_path: Path to save the output GEXF file.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.graph = None

    def load_graph(self):
        """Loads the graph from the specified GraphML file path with error handling."""
        logging.info(f"Attempting to load graph from: {self.input_path}")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file does not exist at {self.input_path}")
            
        try:
            # GraphML preserves structural metadata and attributes cleanly
            self.graph = nx.read_graphml(self.input_path)
            logging.info(f"Successfully loaded graph. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        except Exception as e:
            raise RuntimeError(f"Failed to read GraphML file. Internal error: {str(e)}")

    def compute_k_core_shells(self):
        """
        Computes the K-core shell index for each node and registers it as an attribute.
        Converts directed graphs to undirected if necessary, as standard K-core operates on undirected topologies.
        """
        if self.graph is None:
            raise ValueError("Graph data is not loaded. Cannot compute K-core.")

        logging.info("Computing K-core decomposition shells...")
        
        # Check if the graph is directed and handle appropriately
        analysis_graph = self.graph
        if self.graph.is_directed():
            logging.warning("Input graph is Directed. Converting to Undirected for standard K-core calculation.")
            analysis_graph = self.graph.to_undirected()

        try:
            # nx.core_number returns a dict mapping each node to its highest k-core shell
            core_numbers = nx.core_number(analysis_graph)
            
            # Set the computed values as a node attribute 'k_core_shell'
            nx.set_node_attributes(self.graph, core_numbers, "k_core_shell")
            logging.info("Successfully attached 'k_core_shell' attribute to all nodes.")
        except Exception as e:
            raise RuntimeError(f"Error during K-core decomposition: {str(e)}")

    def export_to_gexf(self):
        """Exports the enriched graph topology into GEXF format for Gephi ingestion."""
        if self.graph is None:
            raise ValueError("Graph data is empty. Cannot export.")

        logging.info(f"Preparing to export enhanced graph structure to: {self.output_path}")
        
        try:
            # Ensure output directory exists before writing
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write out to GEXF layout
            nx.write_gexf(self.graph, self.output_path)
            logging.info(f"Export completed successfully! File saved at: {self.output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to write GEXF file to destination: {str(e)}")


def main():
    """Main execution block parsing line arguments and triggering pipeline execution."""
    # Setting up project structural fallbacks using pathlib
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]  # src/visualization/macro_vis.py -> parents[1] = project root
    
    default_input = project_root / "data" / "processed" / "bitcoin_otc_undirected.graphml"
    default_output = project_root / "data" / "processed" / "bitcoin_otc_kcore_macro.gexf"

    parser = argparse.ArgumentParser(
        description="Calculate K-core shells and generate structural metadata exports for Gephi vĩ mô visualization."
    )
    parser.add_argument(
        "--input_path", 
        type=str, 
        default=str(default_input),
        help="Path to the input graph data file (GraphML format)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=str(default_output),
        help="Destination path for the output processed GEXF file"
    )

    args = parser.parse_args()

    # Executing pipeline operations securely enclosed within error captures
    try:
        visualizer = MacroVisualizer(input_path=args.input_path, output_path=args.output_path)
        visualizer.load_graph()
        visualizer.compute_k_core_shells()
        visualizer.export_to_gexf()
        print("\n--- Pipeline Completed Successfully ---")
    except FileNotFoundError as fnf_err:
        logging.error(f"[File Error] {str(fnf_err)}")
        sys.exit(1)
    except RuntimeError as run_err:
        logging.error(f"[Runtime Exception] {str(run_err)}")
        sys.exit(1)
    except Exception as general_err:
        logging.error(f"[Unexpected System Error] {str(general_err)}")
        sys.exit(1)


if __name__ == "__main__":
    main()