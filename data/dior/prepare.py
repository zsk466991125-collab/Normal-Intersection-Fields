"""Convert native DIOR-R XML and create trainval/test image links."""
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('NIF_DIOR_SOURCE','data/DIOR-R')))
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('NIF_DATA_ROOT','data'))/'dior')
    args = parser.parse_args()
    xml_root = args.source/'Annotations/Oriented Bounding Boxes'
    corners = ('x_left_top','y_left_top','x_right_top','y_right_top',
               'x_right_bottom','y_right_bottom','x_left_bottom','y_left_bottom')
    for split, lists in [('trainval',('train','val')),('test',('test',))]:
        dest = args.output/split
        (dest/'images').mkdir(parents=True, exist_ok=True)
        (dest/'annfiles').mkdir(exist_ok=True)
        ids = [i.strip() for name in lists for i in (args.source/f'ImageSets/Main/{name}.txt').read_text().splitlines() if i.strip()]
        for image_id in ids:
            rows = []
            for obj in ET.parse(xml_root/f'{image_id}.xml').getroot().findall('.//object'):
                box = obj.find('robndbox')
                if box is not None:
                    rows.append(' '.join([box.findtext(k) for k in corners]+[obj.findtext('name'),obj.findtext('difficult','0')]))
            (dest/'annfiles'/f'{image_id}.txt').write_text('\n'.join(rows)+'\n')
            sources = [args.source/folder/f'{image_id}{ext}' for folder in
                       ('JPEGImages','JPEGImages-trainval','JPEGImages-test',f'JPEGImages/{split}')
                       for ext in ('.jpg','.png','.JPG')]
            source = next((p for p in sources if p.is_file()), None)
            if source is None:
                raise FileNotFoundError(f'Missing DIOR image: {image_id}')
            target = dest/'images'/source.name
            if not target.exists():
                target.symlink_to(source.resolve())
        print(f'{split}: {len(ids)} images')


if __name__ == '__main__':
    main()
