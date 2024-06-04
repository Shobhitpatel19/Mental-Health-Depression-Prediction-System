import React from 'react';
import messages from "../HomeContentWidget/messages.json";
import "./style.css";
import {Link} from "react-router-dom";


function NavBar() {
    return (
        <div className='nav-container'>
            <div className="redline"></div>
            <div className="flex-container-with-text-children">
                <div className="navbar-container">
                    <Link to= "/" className="brand-text-heading"> {messages["home"]}</Link>
                    <Link to= "/about" className="heading-text-block" > {messages["about"]}</Link>
                </div>
                <div className="vertical-text-container">
                    <img className="logo" src="./images/3.png" alt="" />
                </div>
                
                <div className="navigation-menu">
                    <div className="contact-profile-box">
                        <Link to="/contactUs" className="brand-text-heading">{messages["contact_us"]}</Link>
                        <Link to="/profile" className="profile-heading">{messages["profile"]}</Link>
                    </div>
                </div>
                
            </div>
        </div>
    )
}

export default NavBar;
