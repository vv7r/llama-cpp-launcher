//! Intègre l'icône et la description dans l'exécutable Windows.
//!
//! L'icône est dessinée par `src/icon.rs`, le même code que l'icône de la
//! fenêtre : le `.ico` est produit ici, dans `OUT_DIR`, rien n'est versionné.

#[path = "src/icon.rs"]
#[allow(dead_code)]
mod icon;

fn main() {
    println!("cargo:rerun-if-changed=src/icon.rs");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ico = out.join("app.ico");
    std::fs::write(&ico, icon::ico()).expect("écriture de app.ico");

    let mut res = winresource::WindowsResource::new();
    res.set_icon(ico.to_str().unwrap())
        .set("FileDescription", "llama.cpp launcher")
        .set("ProductName", "llama.cpp launcher");
    res.compile().expect("compilation des ressources Windows (rc.exe)");
}
