

from Glycan_Formula import COMP_VECT, COMP_COUNT, N_INDEX, parse_composition


SULF_LIM = 3

COMPOSITION_FILE = 'total_overlap_compositions.txt'



COMP_VECT_GLYCRESOFT = ('HexNAc', 'Hex', 'Fuc', 'Neu5Ac', 'Neu5Gc')
COMP_MAP = dict(zip(COMP_VECT, COMP_VECT_GLYCRESOFT))
COMP_ORDER_GLYCRESOFT = ('F', 'H', 'N', 'A', 'G')
COMP_INDICES = dict(zip(COMP_VECT, list(range(COMP_COUNT))))



def convert_to_GlyReSoft_composition(comp_vector: tuple[int, int, int, int, int]) -> str:
    composition_info: list[str] = []
    for comp_symbol in COMP_ORDER_GLYCRESOFT:
        index = COMP_INDICES[comp_symbol]
        count = comp_vector[index]
        if count == 0: continue
        composition_info.append(f'{COMP_MAP[comp_symbol]}:{count}')
    return '; '.join(composition_info)


def build_sulfated_composition_db(db_file_name: str) -> None:
    source_comp_vects: list[tuple[int, int, int, int, int]] = []
    with open(COMPOSITION_FILE, 'r') as file:
        for line in file:
            comp_vect = parse_composition(line.rstrip('\n'))
            source_comp_vects.append(comp_vect)
    sulfated_compositions: list[str] = []
    for comp_vect in source_comp_vects:
        N_count = comp_vect[N_INDEX]
        composition = convert_to_GlyReSoft_composition(comp_vect)
        sulfate_lim = min(N_count, SULF_LIM) + 1
        for i in range(1, sulfate_lim):
            sulfated_compositions.append('{@sulfate:%d; %s}'%(i, composition))
    with open(f'{db_file_name}.txt', 'w') as output:
        for composition in sulfated_compositions:
            output.write(f'{composition}\tN-Glycan\n')




if __name__ == '__main__':

    build_sulfated_composition_db('total_overlap_compositions_sulfate')