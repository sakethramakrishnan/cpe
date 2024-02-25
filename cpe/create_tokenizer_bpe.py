from masked_lm.utils import build_bpe_tokenizer
from masked_lm.dataset import FastaDataset
import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--fasta_path", type=str, required=True)
parser.add_argument("--num_char_per_token", type=int, required=True)
parser.add_argument("--convert_to_aa", type:bool, required=True)
parser.add_argument("--tokenizer_type", type:str, required=True)
parser.add_argument("--vocab_size", type:int, required=True)
parser.add_argument("--save_name", type:str, required=True)
args = parser.parse_args()


dataset = FastaDataset(args.fasta_path, args.num_char_per_token, args.convert_to_aa, args.tokenizer_type)

tokenizer=build_bpe_tokenizer(dataset, args.vocab_size, args.tokenizer_type, save=True, save_name=args.save_name)