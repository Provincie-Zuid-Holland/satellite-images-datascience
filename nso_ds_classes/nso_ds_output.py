import json
import geopandas as gpd


"""

This class handles various methods to write the results to a geojson.

@author: Michael de Winter, Jeroen Esseveld

"""


def produce_geojson(segmentation, fname):
    geojson = '{ "type":"FeatureCollection","crs":{"type":"name","properties":{"name":"epsg:28992"}},"features":['
    for i, seg in enumerate(segmentation):
        rect = [round(seg[0]/2)*2, round(seg[1]/2)*2, 0, 0]
        rect[2], rect[3] = rect[0] + 2, rect[1] + 2
        coords = [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]], [rect[0], rect[1]]]

        geojson += '{"type":"Feature","properties":{"label":"'''+seg[2]+'''"},"geometry":{"type":"Polygon","coordinates":['''+json.dumps(coords)+''']}}'''
        if i+1 != len(segmentation):
            geojson+=","
    geojson +="]}"
    with open(fname, 'w') as f:
        f.write(geojson)

def dissolve_label_geojson(path_in, path_out):
    # open your file with geopandas
    agpd = gpd.GeoDataFrame.from_file(path_in)
    dissolved = gpd.GeoDataFrame(columns=['label', 'geometry'], crs=agpd.crs)
    labels = agpd['label'].unique()
    print("------")
    for label in labels:

        print(label)
        union_gpd = agpd[agpd['label'] == label].unary_union
        dissolved = dissolved.append([{"label":label,"geometry":union_gpd}])
    print("------")

    if '.geojson' not in path_out:
        dissolved.to_file(path_out) 
         
    elif '.geojson' in path_out:
        dissolved.to_file(path_out, driver="GeoJSON")