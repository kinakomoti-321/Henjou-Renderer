#pragma once

#include <map>
#include <renderer/texture.h>
#include <string>

int loadTexture(std::vector<std::shared_ptr<Texture>>& textures, std::map<std::string, int>& known_tex, const std::string& filename, const std::string& modelpath, const std::string& tex_type) {
    if (filename == "") return -1;

    if (known_tex.find(filename) != known_tex.end()) {
        return known_tex[filename];
    }

    int textureID = (int)textures.size();
    textures.push_back(std::make_shared<Texture>(modelpath + "/" + filename, tex_type));
    known_tex[filename] = textureID;

    return textureID;
}
