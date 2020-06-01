###创建本地仓库并与远程仓库关联
* git init
* git add README.md
* git commit -m "first commit"
* git remote add origin https://github.com/EighteenPaint/MLL.git
* git push -u origin master
                
###若已有本地仓库可以直接关联
* git remote add origin https://github.com/EighteenPaint/MLL.git
* git push -u origin master
#关联之后可以使用
* git push origin master（git push 默认master）