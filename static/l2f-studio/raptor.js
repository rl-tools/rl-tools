
const urlParams = new URLSearchParams(window.location.search)
if(["raptor.rl.tools", "ai.new"].includes(window.location.hostname) || urlParams.get("raptor") === "true"){
    
    function handleMobileLayout() {
        const raptorPage = document.getElementById("raptor-project-page");
        const controlsBox = document.getElementById("controls-box");
        const scrollIndicator = document.getElementById("scroll-indicator");
        
        if(window.innerWidth < 768){
            if(raptorPage && controlsBox && raptorPage.parentElement !== controlsBox) {
                if(scrollIndicator && scrollIndicator.nextSibling) {
                    controlsBox.insertBefore(raptorPage, scrollIndicator.nextSibling);
                } else {
                    controlsBox.appendChild(raptorPage);
                }
                raptorPage.classList.remove("raptor-right-sidebar");
                raptorPage.classList.add("controls-container");
            }
        }
        else{
            if(raptorPage && raptorPage.parentElement === controlsBox) {
                document.body.appendChild(raptorPage);
                raptorPage.classList.remove("controls-container");
                raptorPage.classList.add("raptor-right-sidebar");
            }
        }
    }
    
    document.addEventListener("DOMContentLoaded", () => {
        handleMobileLayout();
        if(!urlParams.get("raptor") || urlParams.get("raptor") === "true"){
            document.getElementById("raptor-project-page").style.display = "block";
        }
        if(urlParams.get("yt") !== "false"){
            document.getElementById("video-container-inner").innerHTML = '<iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube-nocookie.com/embed/hVzdWRFTX3k?si=9jCG10p1cvKHO7bP" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>';
        }
        else{
            document.getElementById("video-container").style.display = "none";
        }
    })
    window.addEventListener("resize", handleMobileLayout);
    
}
