function saveXMLFiles( saveXMLPath, resultBox )
tagset = [];
nImg = size( resultBox, 1);
for i= 1:nImg
    tagset.image(i).imageName = resultBox{i,1};
    box = resultBox{i,2};
    nbox = size(box,1);
    tagset.image(i).taggedRectangles.ATTRIBUTE.nTaggedRectangle = nbox;
    for j=1:nbox
        tagset.image(i).taggedRectangles.taggedRectangle(j).ATTRIBUTE.x = box(j,1);
        tagset.image(i).taggedRectangles.taggedRectangle(j).ATTRIBUTE.y = box(j,2);
        tagset.image(i).taggedRectangles.taggedRectangle(j).ATTRIBUTE.width = box(j,3);
        tagset.image(i).taggedRectangles.taggedRectangle(j).ATTRIBUTE.height = box(j,4);
    end
end
Pref=[];
Pref.StructItem = false;
xml_write(saveXMLPath, tagset, 'tagset', Pref);
fprintf( '\n Save to file : %s\n', saveXMLPath );
